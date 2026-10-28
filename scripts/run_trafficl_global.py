from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.yearly_cached_dataset import UnusableYearlyPeriodError, YearlyTrafficStore
from data.window_generator import SlidingWindowGenerator
from models import build_model
from trainers import StreamingContinualTrainer
from trainers.replay_consolidation import consolidate_replay
from train_yearly_continual import make_period_bundle, safe_filename
from utils.checkpointing import atomic_torch_save, try_load_torch, capture_rng_state, restore_rng_state
from utils.config import load_many
from utils.run_provenance import (
    build_run_provenance, prepare_checkpoint_directory,
    validate_checkpoint_provenance, validate_daily_cache,
)


GLOBAL_DATASET_ORDER = [
    "caltrans_bicycle",
    "caltrans_pedestrian",
    "dc_nova",
    "pvcgn_hangzhou",
    "pvcgn_shanghai",
    "shanghai_metroflow",
    "minneapolis_i94",
    "mta_subway",
    "pems03",
]

DAILY_DATASETS = {
    "caltrans_bicycle",
    "caltrans_pedestrian",
    "dc_nova",
    "pvcgn_hangzhou",
    "pvcgn_shanghai",
    "shanghai_metroflow",
}

YEARLY_DATASETS = {
    "minneapolis_i94",
    "mta_subway",
    "pems03",
}

DATASET_CONFIG_MAP = {
    "caltrans_bicycle": "caltrans_bicycle_yearly_full.yaml",
    "caltrans_pedestrian": "caltrans_pedestrian_yearly_full.yaml",
    "dc_nova": "dc_nova_yearly_full.yaml",
    "pvcgn_hangzhou": "pvcgn_hangzhou_yearly_full.yaml",
    "pvcgn_shanghai": "pvcgn_shanghai_yearly_full.yaml",
    "shanghai_metroflow": "shanghai_metroflow_yearly_full.yaml",
    "minneapolis_i94": "minneapolis_i94_yearly_full.yaml",
    "mta_subway": "mta_subway_yearly_full.yaml",
    "pems03": "pems03_full.yaml",
}

CACHE_DIR_CANDIDATES = {
    "caltrans_bicycle": ["nonpems_daily_task_v2/caltrans_bicycle"],
    "caltrans_pedestrian": ["nonpems_daily_task_v2/caltrans_pedestrian"],
    "dc_nova": ["nonpems_daily_task_v2/dc_nova"],
    "pvcgn_hangzhou": ["nonpems_daily_task_v2/pvcgn_hangzhou"],
    "pvcgn_shanghai": ["nonpems_daily_task_v2/pvcgn_shanghai"],
    "shanghai_metroflow": ["nonpems_daily_task_v2/shanghai_metroflow"],
    "minneapolis_i94": [
        "minneapolis_i94_yearly_full",
        "minneapolis_i94",
    ],
    "mta_subway": [
        "mta_subway_yearly_full",
        "mta_subway",
    ],
    "pems03": [
        "pems03_yearly_full",
    ],
}

FORMAL_TASK_COUNTS = {
    "caltrans_bicycle": 365,
    "caltrans_pedestrian": 365,
    "dc_nova": 22,
    "pvcgn_hangzhou": 25,
    "pvcgn_shanghai": 92,
    "shanghai_metroflow": 121,
    "minneapolis_i94": 7,
    "mta_subway": 5,
    "pems03": 18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one persistent TraffiCL model across multiple traffic "
            "datasets without resetting the hierarchical expert banks."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=GLOBAL_DATASET_ORDER,
        help=(
            "Datasets to place in the global continual stream. "
            "Default: all datasets in canonical order."
        ),
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the global stream and print task information "
            "without constructing the model."
        ),
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "models"
            / "trafficl_full.yaml"
        ),
    )

    parser.add_argument(
        "--continual-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "continual"
            / "trafficl_seed42_v5_resume.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--max-global-tasks",
        type=int,
        default=None,
        help="Diagnostic only.",
    )

    parser.add_argument(
        "--max-tasks-per-dataset",
        type=int,
        default=None,
        help=(
            "Diagnostic only. Keep the first N tasks "
            "from every selected dataset."
        ),
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip current-task test evaluation.",
    )

    parser.add_argument("--consolidation-epochs", type=int, default=0,
                        help="Optional final balanced TRAIN replay selected by historical VAL; zero disables it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.consolidation_epochs < 0:
        raise ValueError("--consolidation-epochs cannot be negative")

    dataset_order = (
        list(args.datasets)
        if args.datasets is not None
        else list(GLOBAL_DATASET_ORDER)
    )

    if not dataset_order:
        raise ValueError(
            "global dataset stream is empty"
        )

    if (
        args.max_global_tasks is not None
        and args.max_global_tasks <= 0
    ):
        raise ValueError(
            "--max-global-tasks must be positive"
        )

    if (
        args.max_tasks_per_dataset is not None
        and args.max_tasks_per_dataset <= 0
    ):
        raise ValueError(
            "--max-tasks-per-dataset must be positive"
        )

    set_seed(args.seed)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (
            WORKSPACE_ROOT
            / "results"
            / "trafficl"
            / f"global_state9_seed_{args.seed}"
        )
    )

    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else (
            WORKSPACE_ROOT
            / "03_checkpoints"
            / "trafficl"
            / f"global_state9_seed_{args.seed}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_path = (
        checkpoint_dir
        / "global_stream_state.pt"
    )

    manifest_path = (
        output_dir
        / "global_run_manifest.json"
    )

    assignment_path = (
        output_dir
        / "global_domain_assignment.csv"
    )

    state_routing_path = (
        output_dir
        / "global_state_routing.csv"
    )

    task_metrics_path = (
        output_dir
        / "global_task_metrics.csv"
    )

    eval_history_path = (
            output_dir
            / "continual_eval_history.csv"
    )

    task_level_path = (
            output_dir
            / "task_level_results.csv"
    )

    matrix_mae_path = (
            output_dir
            / "performance_matrix_mae.csv"
    )

    matrix_mse_path = (
            output_dir
            / "performance_matrix_mse.csv"
    )

    matrix_skill_path = (
            output_dir
            / "performance_matrix_skill.csv"
    )

    forgetting_path = (
            output_dir
            / "forgetting_matrix.csv"
    )

    continual_summary_path = (
            output_dir
            / "continual_summary.json"
    )

    global_bundles: list[
        dict[str, Any]
    ] = []

    config_by_dataset: dict[
        str,
        dict[str, Any],
    ] = {}

    metadata_by_dataset: dict[
        str,
        str,
    ] = {}

    skipped: list[
        dict[str, Any]
    ] = []

    for dataset_name in dataset_order:
        dataset_config_path = (
            resolve_dataset_config(
                dataset_name
            )
        )

        config = load_many(
            [
                dataset_config_path,
                args.model_config,
                args.continual_config,
            ]
        )

        config.setdefault(
            "training",
            {},
        )

        config["training"]["seed"] = int(
            args.seed
        )

        validate_trafficl_config(
            config
        )

        metadata_path = (
            resolve_cache_metadata(
                dataset_name,
                args.cache_root,
            )
        )

        validate_task_granularity(
            dataset_name,
            metadata_path,
        )

        store = YearlyTrafficStore(
            metadata_path
        )

        config_by_dataset[
            dataset_name
        ] = config

        metadata_by_dataset[
            dataset_name
        ] = str(
            metadata_path
        )

        window_config = (
            SlidingWindowGenerator
            .from_config(
                config.get(
                    "window",
                    {},
                )
            )
            .config
        )

        experiment_config = dict(
            config.get(
                "experiment",
                {},
            )
        )

        max_windows = optional_int(
            experiment_config.get(
                "max_windows_per_split"
            )
        )

        dataset_bundles: list[
            dict[str, Any]
        ] = []

        for period in store.periods:
            try:
                bundle = make_period_bundle(
                    period,
                    config,
                    window_config,
                    input_columns=(
                        store.input_columns
                    ),
                    target_column=(
                        store.target_column
                    ),
                    max_windows=max_windows,
                )

            except UnusableYearlyPeriodError as exc:
                skipped.append(
                    {
                        "dataset":
                            dataset_name,
                        "task":
                            period.name,
                        "reason":
                            str(exc),
                    }
                )

                print(
                    "[Global Stream] SKIP "
                    f"dataset={dataset_name} "
                    f"task={period.name}: "
                    f"{exc}",
                    flush=True,
                )

                continue

            dataset_bundles.append(
                {
                    **bundle,
                    "dataset_name":
                        dataset_name,
                    "dataset_config":
                        str(
                            dataset_config_path
                        ),
                }
            )

        full_task_count = len(
            dataset_bundles
        )

        expected_task_count = (
            FORMAL_TASK_COUNTS.get(
                dataset_name
            )
        )

        if (
            expected_task_count is not None
            and full_task_count
            != int(
                expected_task_count
            )
        ):
            raise RuntimeError(
                "formal task-count mismatch: "
                f"dataset={dataset_name} "
                f"expected="
                f"{expected_task_count} "
                f"actual="
                f"{full_task_count} "
                f"metadata="
                f"{metadata_path}"
            )

        if (
            args.max_tasks_per_dataset
            is not None
        ):
            dataset_bundles = (
                dataset_bundles[
                    : int(
                        args.max_tasks_per_dataset
                    )
                ]
            )

        if not dataset_bundles:
            raise ValueError(
                f"dataset={dataset_name} "
                "produced no trainable tasks"
            )

        for (
            local_task_index,
            bundle,
        ) in enumerate(
            dataset_bundles
        ):
            bundle[
                "local_task_index"
            ] = int(
                local_task_index
            )

            bundle[
                "global_task_index"
            ] = len(
                global_bundles
            )

            global_bundles.append(
                bundle
            )

        print(
            "[Global Stream] "
            f"dataset={dataset_name} "
            f"tasks="
            f"{len(dataset_bundles)} "
            f"global_tasks_so_far="
            f"{len(global_bundles)}",
            flush=True,
        )

    if (
        args.max_global_tasks
        is not None
    ):
        global_bundles = (
            global_bundles[
                : int(
                    args.max_global_tasks
                )
            ]
        )

    if not global_bundles:
        raise ValueError(
            "no global continual tasks remain"
        )

    for index, bundle in enumerate(
        global_bundles
    ):
        bundle[
            "global_task_index"
        ] = int(index)

    task_names = [
        str(
            bundle[
                "period"
            ].name
        )
        for bundle
        in global_bundles
    ]

    if len(
        set(task_names)
    ) != len(
        task_names
    ):
        duplicates = sorted(
            {
                name
                for name
                in task_names
                if (
                    task_names.count(name)
                    > 1
                )
            }
        )

        raise RuntimeError(
            "global task ids must "
            "be unique; duplicates="
            f"{duplicates}"
        )

    datasets_by_task = {
        str(
            bundle[
                "period"
            ].name
        ):
            bundle[
                "train"
            ]
        for bundle
        in global_bundles
    }

    if args.dry_run:
        counts: dict[
            str,
            int,
        ] = {}

        for bundle in global_bundles:
            name = str(
                bundle[
                    "dataset_name"
                ]
            )

            counts[name] = (
                counts.get(
                    name,
                    0,
                )
                + 1
            )

        print(
            "\n[Global Stream Dry Run]",
            flush=True,
        )

        print(
            f"seed={args.seed}",
            flush=True,
        )

        print(
            f"datasets={dataset_order}",
            flush=True,
        )

        print(
            "task_count_by_dataset="
            f"{counts}",
            flush=True,
        )

        print(
            "global_tasks="
            f"{len(global_bundles)}",
            flush=True,
        )

        print(
            "first_tasks="
            f"{[
                str(bundle['period'].name)
                for bundle
                in global_bundles[:10]
            ]}",
            flush=True,
        )

        print(
            "last_tasks="
            f"{[
                str(bundle['period'].name)
                for bundle
                in global_bundles[-10:]
            ]}",
            flush=True,
        )

        return

    run_provenance = build_run_provenance(
        metadata_by_dataset=metadata_by_dataset,
        config_by_dataset=config_by_dataset,
        task_names=task_names,
        protocol={"runner": "global_continual_stream", "dataset_order": dataset_order,
                  "consolidation_epochs": args.consolidation_epochs},
    )
    prepare_checkpoint_directory(checkpoint_dir, run_provenance, resume=bool(args.resume))

    first_bundle = (
        global_bundles[0]
    )

    first_dataset_name = str(
        first_bundle[
            "dataset_name"
        ]
    )

    base_config = deepcopy(
        config_by_dataset[
            first_dataset_name
        ]
    )

    base_config.setdefault(
        "training",
        {},
    )

    base_config[
        "training"
    ][
        "seed"
    ] = int(
        args.seed
    )

    first_train = (
        first_bundle[
            "train"
        ]
    )

    model = build_model(
        base_config,
        first_train.sample_batch(
            max_windows=2
        ),
    )

    trainer = (
        StreamingContinualTrainer(
            model,
            base_config.get(
                "training",
                {},
            ),
            output_dir=
                output_dir,
            checkpoint_dir=
                checkpoint_dir,
            method=
                "expert_replay_distill",
        )
    )

    trainer.run_provenance = run_provenance
    start_global_task_index = 0

    domain_records: list[
        dict[str, Any]
    ] = []

    state_records: list[
        dict[str, Any]
    ] = []

    task_metric_records: list[
        dict[str, Any]
    ] = []

    num_global_tasks = len(
        global_bundles
    )

    mae_matrix = np.full(
        (
            num_global_tasks,
            num_global_tasks,
        ),
        np.nan,
        dtype=np.float64,
    )

    mse_matrix = np.full(
        (
            num_global_tasks,
            num_global_tasks,
        ),
        np.nan,
        dtype=np.float64,
    )

    skill_matrix = np.full(
        (
            num_global_tasks,
            num_global_tasks,
        ),
        np.nan,
        dtype=np.float64,
    )

    if (
        args.resume
        and state_path.exists()
    ):
        state = try_load_torch(
            state_path,
            map_location=
                trainer.device,
        )

        if state is None:
            raise OSError(
                "global checkpoint "
                "is unreadable: "
                f"{state_path}"
            )

        validate_resume_state(
            state,
            seed=
                args.seed,
            dataset_order=
                dataset_order,
            task_names=
                task_names,
            run_provenance=run_provenance,
        )

        strict_restore_model(
            trainer.model,
            state[
                "model_state"
            ],
        )

        trainer.load_continual_state(
            state[
                "continual_state"
            ],
            datasets_by_task,
        )
        restore_rng_state(state.get("rng_state"))

        start_global_task_index = int(
            state[
                "next_global_task_index"
            ]
        )

        domain_records = list(
            state.get(
                "domain_records",
                [],
            )
        )

        state_records = list(
            state.get(
                "state_records",
                [],
            )
        )

        task_metric_records = list(
            state.get(
                "task_metric_records",
                [],
            )
        )

        restore_evaluation_matrix(
            mae_matrix,
            state.get(
                "evaluation_mae_matrix"
            ),
        )

        restore_evaluation_matrix(
            mse_matrix,
            state.get(
                "evaluation_mse_matrix"
            ),
        )

        restore_evaluation_matrix(
            skill_matrix,
            state.get(
                "evaluation_skill_matrix"
            ),
        )

        print(
            "[Global Resume] "
            "next_global_task_index="
            f"{start_global_task_index} "
            "domains="
            f"{num_domains(trainer.model)} "
            "memory_tasks="
            f"{len(trainer.memory.entries)}",
            flush=True,
        )

    elif args.resume:
        print(
            "[Global Resume] "
            "no checkpoint found at "
            f"{state_path}; "
            "starting clean",
            flush=True,
        )

    prepare_evaluation_history_file(
        eval_history_path,
        committed_task_count=(
            start_global_task_index
        ),
        resume=bool(args.resume),
    )

    prepare_evaluation_history_file(
        task_level_path,
        committed_task_count=(
            start_global_task_index
        ),
        resume=bool(args.resume),
    )

    write_manifest(
        manifest_path,
        seed=
            args.seed,
        dataset_order=
            dataset_order,
        global_bundles=
            global_bundles,
        metadata_by_dataset=
            metadata_by_dataset,
        skipped=
            skipped,
        next_global_task_index=
            start_global_task_index,
        completed=False,
    )

    previous_dataset_name: (
        str | None
    ) = None

    if (
        start_global_task_index
        > 0
    ):
        previous_dataset_name = str(
            global_bundles[
                start_global_task_index
                - 1
            ][
                "dataset_name"
            ]
        )

    for global_task_index in range(
        start_global_task_index,
        len(global_bundles),
    ):
        bundle = global_bundles[
            global_task_index
        ]

        dataset_name = str(
            bundle[
                "dataset_name"
            ]
        )

        task_name = str(
            bundle[
                "period"
            ].name
        )

        local_task_index = int(
            bundle[
                "local_task_index"
            ]
        )

        if (
            dataset_name
            != previous_dataset_name
        ):
            print_dataset_transition(
                previous_dataset_name,
                dataset_name,
                global_task_index,
                trainer,
            )

        domains_before = (
            num_domains(
                trainer.model
            )
        )

        states_before = (
            num_states_per_domain(
                trainer.model
            )
        )

        print(
            "\n"
            + "=" * 76
            + "\n"
            + "[Global Task] "
            + f"{global_task_index + 1}/"
            + f"{len(global_bundles)} "
            + f"dataset={dataset_name} "
            + f"local_task="
            + f"{local_task_index} "
            + f"task={task_name}"
            + "\n"
            + f"domains_before="
            + f"{domains_before} "
            + f"states_before="
            + f"{states_before}"
            + "\n"
            + "=" * 76,
            flush=True,
        )

        fit_summary = trainer.fit_task(
            task_name,
            bundle["train"],
            bundle["val"],
            refit_standardizer=(
                local_task_index
                == 0
            ),
            checkpoint_name=(
                "global_task_"
                f"{global_task_index:04d}_"
                f"{safe_filename(task_name)}.pt"
            ),
        )

        assignment = (
            latest_domain_assignment(
                trainer.model,
                task_name,
            )
        )

        domain_record = {
            "global_task_index":
                int(
                    global_task_index
                ),
            "dataset":
                dataset_name,
            "local_task_index":
                int(
                    local_task_index
                ),
            "task_id":
                task_name,
            "domain_id":
                assignment.get(
                    "domain_id"
                ),
            "domain_action":
                assignment.get(
                    "action"
                ),
            "domain_max_similarity":
                assignment.get(
                    "max_similarity"
                ),
            "domain_similarities":
                json.dumps(
                    assignment.get(
                        "similarities",
                        [],
                    ),
                    ensure_ascii=False,
                ),
            "num_domain_experts":
                num_domains(
                    trainer.model
                ),
            "num_state_experts":
                json.dumps(
                    num_states_per_domain(
                        trainer.model
                    )
                ),
        }

        domain_records.append(
            domain_record
        )

        routing = (
            fit_summary.get(
                "routing"
            )
            if isinstance(
                fit_summary,
                dict,
            )
            else None
        )

        state_routing = (
            routing.get(
                "state_routing"
            )
            if isinstance(
                routing,
                dict,
            )
            else None
        )

        if isinstance(
            state_routing,
            dict,
        ):
            state_records.append(
                {
                    "global_task_index":
                        int(
                            global_task_index
                        ),
                    "dataset":
                        dataset_name,
                    "local_task_index":
                        int(
                            local_task_index
                        ),
                    "task_id":
                        task_name,
                    "domain_id":
                        domain_record[
                            "domain_id"
                        ],
                    "num_state_experts":
                        state_routing.get(
                            "num_state_experts"
                        ),
                    "usage":
                        json.dumps(
                            state_routing.get(
                                "usage",
                                {},
                            ),
                            ensure_ascii=False,
                        ),
                    "actions":
                        json.dumps(
                            state_routing.get(
                                "actions",
                                {},
                            ),
                            ensure_ascii=False,
                        ),
                    "mean_max_similarity":
                        state_routing.get(
                            "mean_max_similarity"
                        ),
                    "min_max_similarity":
                        state_routing.get(
                            "min_max_similarity"
                        ),
                    "max_max_similarity":
                        state_routing.get(
                            "max_max_similarity"
                        ),
                }
            )

        if not args.no_eval:
            print(
                "[Global Evaluation] "
                f"after_task={global_task_index} "
                f"evaluating_tasks=0.."
                f"{global_task_index}",
                flush=True,
            )

            domain_by_task = {
                str(record["task_id"]):
                    record.get("domain_id")
                for record
                in domain_records
            }

            row_records: list[
                dict[str, Any]
            ] = []

            current_task_metrics = None

            for eval_task_index in range(
                global_task_index + 1
            ):
                historical_bundle = (
                    global_bundles[
                        eval_task_index
                    ]
                )

                historical_task_name = str(
                    historical_bundle[
                        "period"
                    ].name
                )

                historical_dataset_name = str(
                    historical_bundle[
                        "dataset_name"
                    ]
                )

                historical_domain_id = (
                    domain_by_task.get(
                        historical_task_name
                    )
                )

                if historical_domain_id is None:
                    raise RuntimeError(
                        "missing historical "
                        "domain assignment: "
                        f"task="
                        f"{historical_task_name}"
                    )

                metrics = trainer.evaluate(
                    historical_bundle[
                        "test"
                    ],
                    task_name=(
                        historical_task_name
                    ),
                    domain_id=int(
                        historical_domain_id
                    ),
                )

                record = {
                    "after_task_index":
                        int(
                            global_task_index
                        ),
                    "after_task":
                        task_name,
                    "eval_task_index":
                        int(
                            eval_task_index
                        ),
                    "eval_task":
                        historical_task_name,
                    "eval_dataset":
                        historical_dataset_name,
                    "domain_id":
                        int(
                            historical_domain_id
                        ),
                }

                for key, value in (
                    metrics.items()
                ):
                    if isinstance(
                        value,
                        (
                            int,
                            float,
                            np.integer,
                            np.floating,
                        ),
                    ):
                        record[key] = float(
                            value
                        )
                    else:
                        record[key] = value

                if "mae" in record:
                    mae_matrix[
                        global_task_index,
                        eval_task_index,
                    ] = float(
                        record["mae"]
                    )

                if "mse" in record:
                    mse_matrix[
                        global_task_index,
                        eval_task_index,
                    ] = float(
                        record["mse"]
                    )

                if "skill" in record:
                    skill_matrix[
                        global_task_index,
                        eval_task_index,
                    ] = float(
                        record["skill"]
                    )

                current_mae = mae_matrix[
                    global_task_index,
                    eval_task_index,
                ]

                if np.isfinite(
                    current_mae
                ):
                    history = mae_matrix[
                        eval_task_index:
                        global_task_index + 1,
                        eval_task_index,
                    ]

                    valid_history = history[
                        np.isfinite(
                            history
                        )
                    ]

                    if valid_history.size:
                        historical_best = float(
                            np.min(
                                valid_history
                            )
                        )

                        record[
                            "historical_best_mae"
                        ] = historical_best

                        record[
                            "forget"
                        ] = max(
                            0.0,
                            float(
                                current_mae
                            )
                            - historical_best,
                        )

                row_records.append(
                    record
                )

                if (
                    eval_task_index
                    == global_task_index
                ):
                    current_task_metrics = {
                        "global_task_index":
                            int(
                                global_task_index
                            ),
                        "dataset":
                            dataset_name,
                        "local_task_index":
                            int(
                                local_task_index
                            ),
                        "task_id":
                            task_name,
                        **{
                            key: value
                            for key, value
                            in record.items()
                            if key not in {
                                "after_task_index",
                                "after_task",
                                "eval_task_index",
                                "eval_task",
                                "eval_dataset",
                                "domain_id",
                                "historical_best_mae",
                                "forget",
                            }
                        },
                    }

            append_records_csv(
                eval_history_path,
                row_records,
            )

            append_records_csv(
                task_level_path,
                row_records,
            )

            if (
                current_task_metrics
                is not None
            ):
                task_metric_records.append(
                    current_task_metrics
                )

        pd.DataFrame(
            domain_records
        ).to_csv(
            assignment_path,
            index=False,
        )

        pd.DataFrame(
            state_records
        ).to_csv(
            state_routing_path,
            index=False,
        )

        if task_metric_records:
            pd.DataFrame(
                task_metric_records
            ).to_csv(
                task_metrics_path,
                index=False,
            )

        save_global_state(
            state_path,
            trainer=trainer,
            seed=args.seed,
            dataset_order=dataset_order,
            task_names=task_names,
            next_global_task_index=(
                global_task_index + 1
            ),
            domain_records=domain_records,
            state_records=state_records,
            task_metric_records=task_metric_records,
            evaluation_mae_matrix=mae_matrix,
            evaluation_mse_matrix=mse_matrix,
            evaluation_skill_matrix=skill_matrix,
        )

        print(
            "[Global Task Complete] "
            f"task={task_name} "
            "domain="
            f"{domain_record['domain_id']} "
            "action="
            f"{domain_record['domain_action']} "
            "domains_after="
            f"{num_domains(trainer.model)} "
            "states_after="
            f"{num_states_per_domain(trainer.model)} "
            "memory_tasks="
            f"{len(trainer.memory.entries)} "
            "memory_by_dataset="
            f"{trainer.memory.task_count_by_dataset()}",
            flush=True,
        )

        previous_dataset_name = (
            dataset_name
        )

        write_manifest(
            manifest_path,
            seed=
                args.seed,
            dataset_order=
                dataset_order,
            global_bundles=
                global_bundles,
            metadata_by_dataset=
                metadata_by_dataset,
            skipped=
                skipped,
            next_global_task_index=
                global_task_index
                + 1,
            completed=False,
        )

    if args.consolidation_epochs:
        consolidation_bundles = [{**bundle, "dataset": bundle["dataset_name"]} for bundle in global_bundles]
        consolidation_routes = [{"domain_id": int(trainer.model.recorded_domain_for_task(bundle["period"].name))}
                                for bundle in global_bundles]
        consolidation = consolidate_replay(trainer, consolidation_bundles, consolidation_routes,
            epochs=args.consolidation_epochs, output_dir=checkpoint_dir / "consolidation")
        (output_dir / "consolidation_summary.json").write_text(json.dumps(consolidation, indent=2), encoding="utf-8")
        if not args.no_eval:
            consolidated_records = []
            for bundle, route in zip(consolidation_bundles, consolidation_routes):
                result = trainer.evaluate(bundle["test"], task_name=bundle["period"].name, domain_id=route["domain_id"])
                consolidated_records.append({"dataset": bundle["dataset"], "task_name": bundle["period"].name,
                    "domain_id": route["domain_id"], "stage": "after_final_train_replay", **result})
            final_frame = pd.DataFrame(consolidated_records)
            final_frame.to_csv(output_dir / "consolidated_final_eval.csv", index=False)
            means = final_frame.groupby("dataset", sort=False)[["mae", "skill"]].mean()
            final_summary = {"stage": "after_final_train_replay",
                "dataset_summary": means.reset_index().to_dict("records"),
                "all_dataset_final_means_positive": bool((means.skill > 0).all())}
            (output_dir / "consolidated_final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    if not args.no_eval:
        write_continual_outputs(
            mae_matrix,
            mse_matrix,
            skill_matrix,
            global_bundles,
            matrix_mae_path=
                matrix_mae_path,
            matrix_mse_path=
                matrix_mse_path,
            matrix_skill_path=
                matrix_skill_path,
            forgetting_path=
                forgetting_path,
            continual_summary_path=
                continual_summary_path,
            seed=
                args.seed,
        )

    write_manifest(
        manifest_path,
        seed=
            args.seed,
        dataset_order=
            dataset_order,
        global_bundles=
            global_bundles,
        metadata_by_dataset=
            metadata_by_dataset,
        skipped=
            skipped,
        next_global_task_index=
            len(
                global_bundles
            ),
        completed=True,
    )

    print(
        "\n[Global TraffiCL Complete] "
        f"seed={args.seed} "
        f"tasks={len(global_bundles)} "
        "domains="
        f"{num_domains(trainer.model)} "
        "states="
        f"{num_states_per_domain(trainer.model)} "
        "memory_tasks="
        f"{len(trainer.memory.entries)}",
        flush=True,
    )


def resolve_dataset_config(
    dataset_name: str,
) -> Path:
    config_dir = (
        PROJECT_ROOT
        / "configs"
        / "datasets"
    )

    filename = (
        DATASET_CONFIG_MAP[
            dataset_name
        ]
    )

    path = (
        config_dir
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            "dataset config is "
            "unavailable for "
            f"{dataset_name}: "
            f"{path}"
        )

    return path


def resolve_cache_metadata(
    dataset: str,
    cache_root: Path,
) -> Path:
    root = (
        cache_root
        .expanduser()
        .resolve()
    )

    candidates = [
        root
        / relative
        / "metadata.json"
        for relative
        in CACHE_DIR_CANDIDATES[
            dataset
        ]
    ]

    for path in candidates:
        if path.exists():
            return path

    tried = "\n  - ".join(
        str(path)
        for path
        in candidates
    )

    raise FileNotFoundError(
        "could not locate formal "
        "cache metadata for "
        f"dataset={dataset!r}. "
        "Tried:\n  - "
        f"{tried}"
    )


def validate_task_granularity(
    dataset: str,
    metadata_path: Path,
) -> None:
    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8-sig"
        )
    )
    validate_daily_cache(metadata, dataset)

    metadata_dataset = str(
        metadata.get(
            "dataset_name",
            "",
        )
    )

    if (
        metadata_dataset
        and metadata_dataset
        != dataset
    ):
        raise ValueError(
            "cache metadata dataset "
            "mismatch: "
            f"runner={dataset!r}, "
            f"metadata="
            f"{metadata_dataset!r}, "
            f"path={metadata_path}"
        )

    periods = list(
        metadata.get(
            "periods",
            [],
        )
    )

    if dataset in DAILY_DATASETS:
        granularity = str(
            metadata.get(
                "task_granularity",
                "",
            )
        ).lower()

        if granularity != "day":
            raise ValueError(
                "formal Global TraffiCL "
                "requires daily-task "
                f"metadata for {dataset}; "
                "task_granularity="
                f"{granularity!r} "
                f"path={metadata_path}"
            )

        if len(periods) <= 1:
            raise ValueError(
                "daily-task cache for "
                f"{dataset} contains only "
                f"{len(periods)} task(s): "
                f"{metadata_path}"
            )

        bad_tasks: list[str] = []

        for period in periods:
            splits = {
                str(
                    shard.get(
                        "split"
                    )
                )
                for shard
                in period.get(
                    "shards",
                    [],
                )
            }

            if splits != {
                "train",
                "val",
                "test",
            }:
                bad_tasks.append(
                    str(
                        period.get(
                            "task_name",
                            period.get(
                                "anchor_date",
                                "unknown",
                            ),
                        )
                    )
                )

                if (
                    len(
                        bad_tasks
                    )
                    >= 3
                ):
                    break

        if bad_tasks:
            raise ValueError(
                "daily-task cache "
                "contains tasks without "
                "explicit train/val/test "
                "shards: "
                f"dataset={dataset} "
                f"examples={bad_tasks}"
            )

    elif dataset in YEARLY_DATASETS:
        granularity = str(
            metadata.get(
                "task_granularity",
                "year",
            )
        ).lower()

        if granularity == "day":
            raise ValueError(
                "formal Global TraffiCL "
                "requires yearly tasks "
                f"for {dataset}, but "
                "daily metadata was "
                "supplied: "
                f"{metadata_path}"
            )


def validate_trafficl_config(
    config: dict[str, Any],
) -> None:
    model_config = dict(
        config.get(
            "model",
            {},
        )
    )

    expected = {
        "max_domain_experts":
            8,
        "max_state_experts_per_domain":
            9,
    }

    for key, value in expected.items():
        actual = (
            model_config.get(
                key
            )
        )

        if actual is None:
            raise KeyError(
                "TraffiCL config "
                f"is missing {key}"
            )

        if int(actual) != int(value):
            raise ValueError(
                "formal TraffiCL "
                f"requires {key}="
                f"{value}, "
                f"got {actual}"
            )

    for key in (
        "domain_similarity_threshold",
        "state_similarity_threshold",
    ):
        actual = float(
            model_config.get(
                key,
                -1.0,
            )
        )

        if not np.isclose(
            actual,
            0.85,
        ):
            raise ValueError(
                "formal TraffiCL "
                f"requires {key}=0.85, "
                f"got {actual}"
            )

    if bool(
        model_config.get(
            "use_dynamic_node_ids",
            False,
        )
    ):
        raise ValueError(
            "formal TraffiCL requires "
            "use_dynamic_node_ids=false"
        )


def strict_restore_model(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
) -> None:
    extra_state = (
        state_dict.get(
            "_extra_state"
        )
    )

    if (
        extra_state is not None
        and hasattr(
            model,
            "set_extra_state",
        )
    ):
        model.set_extra_state(
            extra_state
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )


def latest_domain_assignment(
    model: torch.nn.Module,
    task_name: str,
) -> dict[str, Any]:
    router = getattr(
        model,
        "regime_router",
        None,
    )

    if router is None:
        return {}

    history = getattr(
        router,
        "task_domain_assignment_history",
        [],
    )

    for item in reversed(
        history
    ):
        if str(
            item.get(
                "task_id"
            )
        ) == str(
            task_name
        ):
            return dict(
                item
            )

    return {
        "task_id":
            task_name,
        "domain_id":
            getattr(
                model,
                "active_domain_id",
                None,
            ),
        "action":
            None,
        "similarities":
            [],
        "max_similarity":
            None,
    }


def num_domains(
    model: torch.nn.Module,
) -> int:
    domain_experts = getattr(
        model,
        "domain_experts",
        None,
    )

    if domain_experts is None:
        return 0

    try:
        return int(
            len(
                domain_experts
            )
        )

    except TypeError:
        return 0


def num_states_per_domain(
    model: torch.nn.Module,
) -> list[int]:
    banks = getattr(
        model,
        "state_expert_banks",
        None,
    )

    if banks is None:
        return []

    return [
        int(
            len(bank)
        )
        for bank
        in banks
    ]


def print_dataset_transition(
    previous_dataset:
        str | None,
    current_dataset:
        str,
    global_task_index:
        int,
    trainer:
        StreamingContinualTrainer,
) -> None:
    print(
        "\n[Dataset Transition] "
        f"from={previous_dataset} "
        f"to={current_dataset} "
        "global_task_index="
        f"{global_task_index} "
        "preserve_model=True "
        "preserve_domain_bank=True "
        "preserve_state_bank=True "
        "preserve_memory=True "
        "preserve_teacher=True "
        "domains_before_transition="
        f"{num_domains(trainer.model)} "
        "states_before_transition="
        f"{num_states_per_domain(trainer.model)} "
        "memory_tasks="
        f"{len(trainer.memory.entries)} "
        "memory_by_dataset="
        f"{trainer.memory.task_count_by_dataset()}",
        flush=True,
    )


def save_global_state(
    path: Path,
    *,
    trainer:
        StreamingContinualTrainer,
    seed: int,
    dataset_order:
        list[str],
    task_names:
        list[str],
    next_global_task_index:
        int,
    domain_records:
        list[dict[str, Any]],
    state_records:
        list[dict[str, Any]],
    task_metric_records:
        list[dict[str, Any]],
    evaluation_mae_matrix:
        np.ndarray,
    evaluation_mse_matrix:
        np.ndarray,
    evaluation_skill_matrix:
        np.ndarray,
) -> None:
    completed = int(
        next_global_task_index
    )

    atomic_torch_save(
        {
            "protocol":
                "global_continual_stream",
            "rng_state": capture_rng_state(),
            "run_provenance": trainer.run_provenance,
            "seed":
                int(seed),
            "dataset_order":
                list(
                    dataset_order
                ),
            "task_names":
                list(
                    task_names
                ),
            "next_global_task_index":
                completed,
            "model_state":
                trainer.model.state_dict(),
            "continual_state":
                trainer.continual_state_dict(),
            "domain_records":
                list(
                    domain_records
                ),
            "state_records":
                list(
                    state_records
                ),
            "task_metric_records":
                list(
                    task_metric_records
                ),
            # Only the committed lower-left square is checkpointed.
            # This keeps early checkpoints small while remaining
            # resume-safe for the full 1020-task stream.
            "evaluation_mae_matrix":
                np.asarray(
                    evaluation_mae_matrix[
                        :completed,
                        :completed,
                    ],
                    dtype=np.float64,
                ),
            "evaluation_mse_matrix":
                np.asarray(
                    evaluation_mse_matrix[
                        :completed,
                        :completed,
                    ],
                    dtype=np.float64,
                ),
            "evaluation_skill_matrix":
                np.asarray(
                    evaluation_skill_matrix[
                        :completed,
                        :completed,
                    ],
                    dtype=np.float64,
                ),
        },
        path,
    )


def validate_resume_state(
    state: dict[str, Any],
    *,
    seed: int,
    dataset_order:
        list[str],
    task_names:
        list[str],
    run_provenance: dict[str, Any],
) -> None:
    validate_checkpoint_provenance(state, run_provenance)
    if (
        state.get(
            "protocol"
        )
        != "global_continual_stream"
    ):
        raise RuntimeError(
            "refusing to resume "
            "from a non-global "
            "TraffiCL checkpoint"
        )

    if int(
        state.get(
            "seed",
            -1,
        )
    ) != int(
        seed
    ):
        raise RuntimeError(
            "resume seed mismatch: "
            "checkpoint="
            f"{state.get('seed')} "
            f"requested={seed}"
        )

    saved_order = list(
        state.get(
            "dataset_order",
            [],
        )
    )

    if saved_order != list(
        dataset_order
    ):
        raise RuntimeError(
            "resume dataset order "
            "mismatch: "
            f"checkpoint="
            f"{saved_order}, "
            f"requested="
            f"{dataset_order}"
        )

    saved_tasks = list(
        state.get(
            "task_names",
            [],
        )
    )

    if saved_tasks != list(
        task_names
    ):
        raise RuntimeError(
            "resume global task "
            "stream does not match "
            "the current cache/config"
        )


def write_manifest(
    path: Path,
    *,
    seed: int,
    dataset_order:
        list[str],
    global_bundles:
        list[dict[str, Any]],
    metadata_by_dataset:
        dict[str, str],
    skipped:
        list[dict[str, Any]],
    next_global_task_index:
        int,
    completed: bool,
) -> None:
    task_count_by_dataset: dict[
        str,
        int,
    ] = {}

    for bundle in global_bundles:
        dataset_name = str(
            bundle[
                "dataset_name"
            ]
        )

        task_count_by_dataset[
            dataset_name
        ] = (
            task_count_by_dataset.get(
                dataset_name,
                0,
            )
            + 1
        )

    payload = {
        "model_name":
            "TraffiCL",
        "protocol":
            "global_continual_stream",
        "evaluation_policy":
            "full_lower_triangular_normalized_mae_mse",
        "seed":
            int(seed),
        "dataset_order":
            list(
                dataset_order
            ),
        "state_cap":
            9,
        "domain_cap":
            8,
        "domain_threshold":
            0.85,
        "state_threshold":
            0.85,
        "global_num_tasks":
            int(
                len(
                    global_bundles
                )
            ),
        "global_num_tasks_expected_full":
            int(
                sum(
                    FORMAL_TASK_COUNTS.values()
                )
            ),
        "task_count_by_dataset":
            task_count_by_dataset,
        "metadata_by_dataset":
            metadata_by_dataset,
        "next_global_task_index":
            int(
                next_global_task_index
            ),
        "completed":
            bool(
                completed
            ),
        "skipped_tasks":
            skipped,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )



def restore_evaluation_matrix(
    target: np.ndarray,
    saved: Any,
) -> None:
    if saved is None:
        return

    source = np.asarray(
        saved,
        dtype=np.float64,
    )

    if source.ndim != 2:
        raise RuntimeError(
            "saved evaluation matrix "
            f"must be 2-D, got "
            f"shape={source.shape}"
        )

    rows = min(
        target.shape[0],
        source.shape[0],
    )

    cols = min(
        target.shape[1],
        source.shape[1],
    )

    target[
        :rows,
        :cols,
    ] = source[
        :rows,
        :cols,
    ]


def prepare_evaluation_history_file(
    path: Path,
    *,
    committed_task_count: int,
    resume: bool,
) -> None:
    """
    Keep only rows already committed by the global checkpoint.

    If an interruption happened after CSV writing but before the global
    checkpoint was atomically committed, rows from that unfinished task
    are removed here before resumed evaluation continues.
    """
    if not path.exists():
        return

    if not resume:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    try:
        frame = pd.read_csv(
            path
        )
    except pd.errors.EmptyDataError:
        return

    if (
        "after_task_index"
        not in frame.columns
    ):
        raise RuntimeError(
            "evaluation history is missing "
            f"after_task_index: {path}"
        )

    frame = frame[
        frame[
            "after_task_index"
        ].astype(int)
        < int(
            committed_task_count
        )
    ]

    frame.to_csv(
        path,
        index=False,
    )


def append_records_csv(
    path: Path,
    records: list[
        dict[str, Any]
    ],
) -> None:
    if not records:
        return

    frame = pd.DataFrame(
        records
    )

    write_header = (
        not path.exists()
        or path.stat().st_size == 0
    )

    frame.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
    )


def _forgetting_matrix(
    performance_matrix: np.ndarray,
) -> np.ndarray:
    n_tasks = int(
        performance_matrix.shape[0]
    )

    forgetting = np.full(
        performance_matrix.shape,
        np.nan,
        dtype=np.float64,
    )

    for i in range(n_tasks):
        for j in range(i + 1):
            current = (
                performance_matrix[
                    i,
                    j,
                ]
            )

            if not np.isfinite(
                current
            ):
                continue

            history = (
                performance_matrix[
                    j : i + 1,
                    j,
                ]
            )

            valid = history[
                np.isfinite(
                    history
                )
            ]

            if valid.size == 0:
                continue

            best = float(
                np.min(valid)
            )

            forgetting[
                i,
                j,
            ] = max(
                0.0,
                float(current) - best,
            )

    return forgetting


def write_continual_outputs(
    mae_matrix: np.ndarray,
    mse_matrix: np.ndarray,
    skill_matrix: np.ndarray,
    global_bundles:
        list[dict[str, Any]],
    *,
    matrix_mae_path: Path,
    matrix_mse_path: Path,
    matrix_skill_path: Path,
    forgetting_path: Path,
    continual_summary_path: Path,
    seed: int,
) -> None:
    task_names = [
        str(
            bundle[
                "period"
            ].name
        )
        for bundle
        in global_bundles
    ]

    dataset_names = [
        str(
            bundle[
                "dataset_name"
            ]
        )
        for bundle
        in global_bundles
    ]

    pd.DataFrame(
        mae_matrix,
        index=task_names,
        columns=task_names,
    ).to_csv(
        matrix_mae_path,
        index_label="after_task",
    )

    pd.DataFrame(
        mse_matrix,
        index=task_names,
        columns=task_names,
    ).to_csv(
        matrix_mse_path,
        index_label="after_task",
    )

    pd.DataFrame(
        skill_matrix,
        index=task_names,
        columns=task_names,
    ).to_csv(
        matrix_skill_path,
        index_label="after_task",
    )

    forgetting_mae = (
        _forgetting_matrix(
            mae_matrix
        )
    )

    forgetting_mse = (
        _forgetting_matrix(
            mse_matrix
        )
    )

    pd.DataFrame(
        forgetting_mae,
        index=task_names,
        columns=task_names,
    ).to_csv(
        forgetting_path,
        index_label="after_task",
    )

    completed_rows = [
        i
        for i in range(
            len(task_names)
        )
        if np.isfinite(
            mae_matrix[i, i]
        )
    ]

    summary: dict[str, Any] = {
        "model":
            "TraffiCL",
        "protocol":
            "global_continual_stream",
        "evaluation_policy":
            (
                "full_lower_triangular_"
                "normalized_mae_mse"
            ),
        "seed":
            int(seed),
        "num_tasks_total":
            int(
                len(task_names)
            ),
        "num_tasks_evaluated":
            int(
                len(
                    completed_rows
                )
            ),
    }

    if completed_rows:
        last_i = int(
            completed_rows[-1]
        )

        final_mae = (
            mae_matrix[
                last_i,
                : last_i + 1,
            ]
        )

        final_mse = (
            mse_matrix[
                last_i,
                : last_i + 1,
            ]
        )

        final_skill = (
            skill_matrix[
                last_i,
                : last_i + 1,
            ]
        )

        valid_final_mae = (
            final_mae[
                np.isfinite(
                    final_mae
                )
            ]
        )

        valid_final_mse = (
            final_mse[
                np.isfinite(
                    final_mse
                )
            ]
        )

        valid_final_skill = (
            final_skill[
                np.isfinite(
                    final_skill
                )
            ]
        )

        diagonal_mae = np.asarray(
            [
                mae_matrix[i, i]
                for i
                in completed_rows
                if np.isfinite(
                    mae_matrix[i, i]
                )
            ],
            dtype=np.float64,
        )

        final_forget_mae = [
            float(
                forgetting_mae[
                    last_i,
                    j,
                ]
            )
            for j
            in range(last_i)
            if np.isfinite(
                forgetting_mae[
                    last_i,
                    j,
                ]
            )
        ]

        final_forget_mse = [
            float(
                forgetting_mse[
                    last_i,
                    j,
                ]
            )
            for j
            in range(last_i)
            if np.isfinite(
                forgetting_mse[
                    last_i,
                    j,
                ]
            )
        ]

        bwt_mae_values = []
        bwt_mse_values = []

        for j in range(last_i):
            if (
                np.isfinite(
                    mae_matrix[j, j]
                )
                and np.isfinite(
                    mae_matrix[
                        last_i,
                        j,
                    ]
                )
            ):
                bwt_mae_values.append(
                    float(
                        mae_matrix[j, j]
                        - mae_matrix[
                            last_i,
                            j,
                        ]
                    )
                )

            if (
                np.isfinite(
                    mse_matrix[j, j]
                )
                and np.isfinite(
                    mse_matrix[
                        last_i,
                        j,
                    ]
                )
            ):
                bwt_mse_values.append(
                    float(
                        mse_matrix[j, j]
                        - mse_matrix[
                            last_i,
                            j,
                        ]
                    )
                )

        summary.update(
            {
                "final_task_index":
                    last_i,
                "final_task":
                    task_names[
                        last_i
                    ],
                "final_average_performance":
                    (
                        float(
                            np.mean(
                                valid_final_mae
                            )
                        )
                        if valid_final_mae.size
                        else None
                    ),
                "final_average_mae":
                    (
                        float(
                            np.mean(
                                valid_final_mae
                            )
                        )
                        if valid_final_mae.size
                        else None
                    ),
                "final_average_mse":
                    (
                        float(
                            np.mean(
                                valid_final_mse
                            )
                        )
                        if valid_final_mse.size
                        else None
                    ),
                "final_average_skill":
                    (
                        float(
                            np.mean(
                                valid_final_skill
                            )
                        )
                        if valid_final_skill.size
                        else None
                    ),
                "average_new_task_performance":
                    (
                        float(
                            np.mean(
                                diagonal_mae
                            )
                        )
                        if diagonal_mae.size
                        else None
                    ),
                "avg_forget_mae":
                    (
                        float(
                            np.mean(
                                final_forget_mae
                            )
                        )
                        if final_forget_mae
                        else 0.0
                    ),
                "max_forget_mae":
                    (
                        float(
                            np.max(
                                final_forget_mae
                            )
                        )
                        if final_forget_mae
                        else 0.0
                    ),
                "avg_forget_mse":
                    (
                        float(
                            np.mean(
                                final_forget_mse
                            )
                        )
                        if final_forget_mse
                        else 0.0
                    ),
                "max_forget_mse":
                    (
                        float(
                            np.max(
                                final_forget_mse
                            )
                        )
                        if final_forget_mse
                        else 0.0
                    ),
                "bwt_mae":
                    (
                        float(
                            np.mean(
                                bwt_mae_values
                            )
                        )
                        if bwt_mae_values
                        else 0.0
                    ),
                "bwt_mse":
                    (
                        float(
                            np.mean(
                                bwt_mse_values
                            )
                        )
                        if bwt_mse_values
                        else 0.0
                    ),
            }
        )

        summary[
            "avg_forgetting"
        ] = summary[
            "avg_forget_mae"
        ]

        summary[
            "max_forgetting"
        ] = summary[
            "max_forget_mae"
        ]

        summary[
            "backward_transfer"
        ] = summary[
            "bwt_mae"
        ]

        per_dataset_final: dict[
            str,
            dict[str, Any],
        ] = {}

        for dataset_name in (
            dict.fromkeys(
                dataset_names
            )
        ):
            indices = [
                index
                for index, name
                in enumerate(
                    dataset_names
                )
                if (
                    name
                    == dataset_name
                    and index <= last_i
                )
            ]

            dataset_mae = [
                float(
                    mae_matrix[
                        last_i,
                        index,
                    ]
                )
                for index
                in indices
                if np.isfinite(
                    mae_matrix[
                        last_i,
                        index,
                    ]
                )
            ]

            dataset_mse = [
                float(
                    mse_matrix[
                        last_i,
                        index,
                    ]
                )
                for index
                in indices
                if np.isfinite(
                    mse_matrix[
                        last_i,
                        index,
                    ]
                )
            ]

            dataset_skill = [
                float(
                    skill_matrix[
                        last_i,
                        index,
                    ]
                )
                for index
                in indices
                if np.isfinite(
                    skill_matrix[
                        last_i,
                        index,
                    ]
                )
            ]

            if not dataset_mae:
                continue

            per_dataset_final[
                dataset_name
            ] = {
                "num_tasks":
                    int(
                        len(
                            dataset_mae
                        )
                    ),
                "mae":
                    float(
                        np.mean(
                            dataset_mae
                        )
                    ),
                "mse":
                    (
                        float(
                            np.mean(
                                dataset_mse
                            )
                        )
                        if dataset_mse
                        else None
                    ),
                "skill":
                    (
                        float(
                            np.mean(
                                dataset_skill
                            )
                        )
                        if dataset_skill
                        else None
                    ),
            }

        summary[
            "per_dataset_final"
        ] = per_dataset_final

    continual_summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

def optional_int(
    value: Any,
) -> int | None:
    if value in (
        None,
        "",
    ):
        return None

    return int(
        value
    )


def set_seed(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


if __name__ == "__main__":
    main()
