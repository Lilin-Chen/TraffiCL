# Reproducibility and release details

**TraffiCL: Regime-Aware Continual Learning for Traffic Forecasting**

This model-only release packages the V5 implementation used by the completed eight-dataset,
1,002-task experiments. It is grounded in recorded runs rather than the legacy
README or the names of experimental configurations. The 19 implementation files
in the original archive matched all five recorded runs (seeds 42–46).
This reduced release keeps the V5 model and training computation unchanged,
removes unrelated model registration branches, and trims package imports.
`reproducibility/source_scope.json` records the distinction. Run
`python scripts/verify_release.py` to verify the retained unchanged files.

## Model and training configuration

The registered model is `expert_lite_inverted_traffic`, implemented by
`ExpertLiteInvertedTrafficTransformer` in
`models/backbones/expert_lightweight_transformer.py`.

The model uses flow channel 0, per-window normalization, full-history node-token
projection, and a cross-node Transformer. Hierarchical domain and state adapters
share the backbone. The recorded model has width 64, four attention heads, two
layers, feedforward width 128, dropout 0.1, and adapter bottleneck 16. Expert
limits are eight domains and nine states per domain; both similarity thresholds
are 0.85. Dynamic node IDs are disabled.

Use `configs/models/trafficl_full.yaml` with
`configs/continual/trafficl_seed42_v5_resume.yaml`. The latter filename contains
`resume`, but is also the verified configuration for a fresh run; the `--resume`
flag separately controls checkpoint recovery. The recorded training uses MAE,
direct targets, Adam settings supplied by the trainer, learning rate 0.001,
batch size 64, up to 20 epochs per task, patience 5, 64 replay examples per task,
replay weight 0.5, distillation weight 0.1, and teacher EMA decay 0.99.
The first task's training statistics are retained for each dataset.

Only TraffiCL V5 is registered. Baseline experiments, ablations, sensitivity
sweeps, diagnostics, other model configurations, and their runners are excluded.
The original shared backbone support and trainer files are retained because V5
depends on them. The persistence calculation used by Skill remains a metric
dependency. The global runner also retains optional PeMS cache support;
the paper protocol below explicitly selects the eight verified datasets.

## Installation

Create a Python environment, install a PyTorch build suitable for your hardware,
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

Dependencies are unpinned because the archive does not contain a verified lockfile
for the full runs. The original environment and hardware are not reconstructed
by this packaging step; identical numerical results on other systems are not
guaranteed.

## Data and task stream

Raw data, prepared caches, trained weights, large result matrices, logs, local
documents, obsolete backup files, and nested historical archives are excluded.
Supply the existing prepared cache through `--cache-root`. Dataset source paths
in the released configs are portable `data/raw/...` placeholders; adapt those
paths before rebuilding data. Numerical model and training settings are retained.

| Dataset | Tasks | Cache directory under CACHE_ROOT |
| --- | ---: | --- |
| Caltrans Bicycle | 365 | nonpems_daily_task_v2/caltrans_bicycle |
| Caltrans Pedestrian | 365 | nonpems_daily_task_v2/caltrans_pedestrian |
| DC Nova | 22 | nonpems_daily_task_v2/dc_nova |
| PVCGN Hangzhou | 25 | nonpems_daily_task_v2/pvcgn_hangzhou |
| PVCGN Shanghai | 92 | nonpems_daily_task_v2/pvcgn_shanghai |
| Shanghai MetroFlow | 121 | nonpems_daily_task_v2/shanghai_metroflow |
| Minneapolis I94 | 7 | minneapolis_i94_yearly_full |
| MTA Subway | 5 | mta_subway_yearly_full |

Each directory must contain `metadata.json` and its referenced shards. Daily
caches must declare `target_disjoint_daily_v2`; cross-boundary targets are masked.
The model and persistence baseline use the same valid-target mask. DC Nova uses
synthetic sample-order timestamps; this does not establish real calendar-time
separation. This model-only release expects the existing verified caches. Cache builders
and data-audit experiment utilities are excluded. It does not provide a
one-command download of the raw datasets.

## Reproduce the recorded protocol

Run from the repository root. Replace `/path/to/cache` with your prepared cache.
First append `--dry-run` to the training command to validate the stream.

```bash
python scripts/run_trafficl_global.py --cache-root /path/to/cache --model-config configs/models/trafficl_full.yaml --continual-config configs/continual/trafficl_seed42_v5_resume.yaml --seed 42 --datasets caltrans_bicycle caltrans_pedestrian dc_nova pvcgn_hangzhou pvcgn_shanghai shanghai_metroflow minneapolis_i94 mta_subway --consolidation-epochs 20 --output-dir results/v5_seed42 --checkpoint-dir results/v5_seed42/checkpoints
```

For seeds 43, 44, 45, and 46, change `--seed`, choose a separate output and
checkpoint directory, and set `--consolidation-epochs 0`. Each experiment trains
one persistent model across the complete stream, retaining its expert banks.
To continue an interrupted run, use the same command and directories with
`--resume`. Checkpoint provenance rejects incompatible settings or cache content.
The global runner now defaults to the verified V5 training configuration.
The reduced source set creates a new provenance identity: use `--resume` for runs
started with this release. Historical checkpoint directories may be rejected;
do not overwrite their provenance to bypass this check.

Seed 42 records 20 optional consolidation epochs after the stream; seeds 43–46
record zero. The final continual matrix row and `consolidated_final_eval.csv`
represent different evaluation stages. Do not combine them as equivalent runs.


## Evaluation and paper alignment

Inspect `task_level_results.csv`, `continual_eval_history.csv`,
`global_task_metrics.csv`, performance matrices, and the run manifest in the
chosen output directory. Output filenames and the manifest's historical
`evaluation_policy` label alone do not establish metric normalization.

The V5 global evaluation stores MAE and MSE in original target units. Paper
normalized MAE/MSE require explicit task-wise division by the test-target standard
deviation/variance and task macro averaging. This packaging step does not change
the evaluator or silently normalize its outputs.

Skill is `1 - raw_MAE / persistence_MAE` on the same mask. A persistence MAE at
or below 1e-8 produces an undefined Skill. Forgetting uses current error minus
the minimum historical error including the current stage; improvement therefore
does not produce negative forgetting. BWT uses initial error minus final error,
so positive values indicate improvement. Specify the error scale and task set
when reporting either metric. Historical evaluation restores the domain recorded
for each task; the complete protocol should not be described as task-ID-free.

The checked local manuscript (`TraffiCL_V5_revised/main.tex`, identified by its
SHA-256 in `reproducibility/final_audit.json`) explicitly reports seed 42 at the
end of the 1,002-task continual stream, excluding optional consolidation. Its
architecture and training settings match the recorded V5 configuration. These
are single-seed table results, not five-seed averages. Included run metadata
establishes model identity and completion, not an independent rerun of training.

## Repository layout

- `models/`: the V5 model, its shared backbone support, and a V5-only registry.
- `trainers/`: required V5 training and consolidation components.
- `data/`, `utils/`: required loading and runtime dependencies.
- `configs/`: the V5 model/training settings and dataset loading configurations.
- `scripts/run_trafficl_global.py`: the single training entry point.
- `scripts/verify_release.py`: retained implementation verification.
- `reproducibility/`: run identity evidence and release scope.

## Release scope and licensing

All published documentation is English. The model backbone and training computation
files are byte-for-byte unchanged. Registry/import-only changes and removed
files are listed in `reproducibility/source_scope.json`. Machine-specific dataset configuration paths
were relocated; original cache fingerprints remain in the provenance records.
New runs may receive different provenance identities after relocation.

Two unrelated legacy utilities containing local Chinese paths and messages
(`data/build_pems03_npz.py` and `scripts/check_project.py`) are omitted. Old Chinese
handoff documents and verification reports are replaced by this README. Original
files are preserved in the input archive.

The author has selected the MIT License for this release. See the repository's
LICENSE file for its terms. Citation author details, publication venue, and
artifact URL are not invented in this release.
