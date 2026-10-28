from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from trainers.streaming_forecasting_trainer import (
    StreamingForecastingTrainer,
    unpack_batch,
    unpack_sample,
)
from utils.checkpointing import atomic_torch_save, try_load_torch, capture_rng_state, restore_rng_state
from utils.progress import LogProgressBar


SUPPORTED_METHODS = {
    "sequential",
    "replay",
    "ewc",
    "lwf",
    "derpp",
    "expert_replay_distill",
}


@dataclass
class ReplayEntry:
    name: str
    dataset: Dataset

    # Global continual protocol:
    # remember which physical dataset this task belongs to.
    dataset_name: str | None

    indices: np.ndarray
    logits: np.ndarray | None = None
    anchor_pred: np.ndarray | None = None
    domain_id: int | None = None
    state_ids: np.ndarray | None = None


@dataclass
class ReplayGroup:
    entry: ReplayEntry
    positions: np.ndarray
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    logits: np.ndarray | None = None
    anchor_pred: np.ndarray | None = None
    state_ids: np.ndarray | None = None

    @property
    def task_name(self) -> str:
        return self.entry.name

    @property
    def dataset_name(self) -> str | None:
        return self.entry.dataset_name

    @property
    def domain_id(self) -> int | None:
        return self.entry.domain_id

    @property
    def sample_count(self) -> int:
        return int(len(self.positions))


@dataclass
class StreamingReplayMemory:
    capacity_per_task: int = 64
    seed: int = 42
    entries: list[ReplayEntry] = field(default_factory=list)
    balanced_replay_cursor: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def add(
            self,
            name: str,
            dataset: Dataset,
            *,
            dataset_name: str | None = None,
    ) -> ReplayEntry | None:
        if self.capacity_per_task <= 0 or len(dataset) == 0:
            return None

        count = min(
            int(self.capacity_per_task),
            len(dataset),
        )

        indices = np.sort(
            self.rng.choice(
                len(dataset),
                size=count,
                replace=False,
            )
        )

        entry = ReplayEntry(
            name=name,
            dataset=dataset,
            dataset_name=dataset_name,
            indices=indices.astype(np.int64),
        )

        self.entries.append(entry)
        return entry

    def sample(
        self,
        batch_size: int,
    ) -> tuple[ReplayEntry, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
        if not self.entries:
            return None
        entry = self.entries[int(self.rng.integers(0, len(self.entries)))]
        count = min(int(batch_size), len(entry.indices))
        positions = self.rng.choice(len(entry.indices), size=count, replace=False)
        xs, ys, masks = [], [], []
        for position in positions:
            x, y, mask = unpack_sample(entry.dataset[int(entry.indices[position])])
            xs.append(x)
            ys.append(y)
            masks.append(mask)
        logits = None if entry.logits is None else entry.logits[positions]
        return (
            entry,
            np.stack(xs).astype(np.float32),
            np.stack(ys).astype(np.float32),
            np.stack(masks).astype(np.float32),
            logits,
        )

    def sample_balanced(self, batch_size: int) -> list[ReplayGroup]:
        if not self.entries or batch_size <= 0:
            return []
        total = int(batch_size)
        task_count = len(self.entries)
        if task_count <= total:
            base_quota = total // task_count
            remainder = total % task_count
            quotas = [base_quota for _ in self.entries]
            for offset in range(remainder):
                quotas[(self.balanced_replay_cursor + offset) % task_count] += 1
            self.balanced_replay_cursor = (self.balanced_replay_cursor + remainder) % task_count
            selected = [(index, quota) for index, quota in enumerate(quotas) if quota > 0]
        else:
            selected = [
                ((self.balanced_replay_cursor + offset) % task_count, 1)
                for offset in range(total)
            ]
            self.balanced_replay_cursor = (self.balanced_replay_cursor + total) % task_count
        return [
            group
            for index, quota in selected
            if (group := self._sample_entry_group(self.entries[index], quota)) is not None
        ]

    def _sample_entry_group(self, entry: ReplayEntry, quota: int) -> ReplayGroup | None:
        if quota <= 0 or len(entry.indices) == 0:
            return None
        replace = int(quota) > len(entry.indices)
        positions = self.rng.choice(len(entry.indices), size=int(quota), replace=replace)
        xs, ys, masks = [], [], []
        for position in positions:
            x, y, mask = unpack_sample(entry.dataset[int(entry.indices[int(position)])])
            xs.append(x)
            ys.append(y)
            masks.append(mask)
        logits = None if entry.logits is None else entry.logits[positions]
        anchor_pred = None if entry.anchor_pred is None else entry.anchor_pred[positions]
        state_ids = None if entry.state_ids is None else entry.state_ids[positions]
        return ReplayGroup(
            entry=entry,
            positions=np.asarray(positions, dtype=np.int64),
            x=np.stack(xs).astype(np.float32),
            y=np.stack(ys).astype(np.float32),
            mask=np.stack(masks).astype(np.float32),
            logits=None if logits is None else np.asarray(logits, dtype=np.float32),
            anchor_pred=None if anchor_pred is None else np.asarray(anchor_pred, dtype=np.float32),
            state_ids=None if state_ids is None else np.asarray(state_ids, dtype=np.int64),
        )

    @property
    def size(self) -> int:
        return int(
            sum(
                len(entry.indices)
                for entry in self.entries
            )
        )

    def size_by_task(
            self,
    ) -> dict[str, int]:
        return {
            entry.name: int(len(entry.indices))
            for entry in self.entries
        }

    def size_by_dataset(
            self,
    ) -> dict[str, int]:
        result: dict[str, int] = {}

        for entry in self.entries:
            key = (
                entry.dataset_name
                if entry.dataset_name is not None
                else "UNKNOWN"
            )

            result[key] = (
                    result.get(key, 0)
                    + int(len(entry.indices))
            )

        return result

    def task_count_by_dataset(
            self,
    ) -> dict[str, int]:
        result: dict[str, int] = {}

        for entry in self.entries:
            key = (
                entry.dataset_name
                if entry.dataset_name is not None
                else "UNKNOWN"
            )

            result[key] = (
                    result.get(key, 0)
                    + 1
            )

        return result


class TeacherGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass
class OnlineEWCState:
    fisher: dict[str, torch.Tensor] = field(default_factory=dict)
    means: dict[str, torch.Tensor] = field(default_factory=dict)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        result = None
        for name, parameter in model.named_parameters():
            if name not in self.fisher or name not in self.means:
                continue
            term = (self.fisher[name] * (parameter - self.means[name]).square()).sum()
            result = term if result is None else result + term
        if result is None:
            return next(model.parameters()).new_zeros(())
        return result

    def consolidate(
        self,
        model: nn.Module,
        new_fisher: Mapping[str, torch.Tensor],
        *,
        decay: float,
    ) -> None:
        for name, parameter in model.named_parameters():
            if name not in new_fisher:
                continue
            current = new_fisher[name].detach().clone()
            if name in self.fisher:
                current = float(decay) * self.fisher[name] + current
            self.fisher[name] = current
            self.means[name] = parameter.detach().clone()


class StreamingContinualTrainer:
    """Continual methods for yearly datasets whose station dimensions can differ."""

    def __init__(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
        *,
        output_dir: str | Path,
        method: str,
        checkpoint_dir: str | Path | None = None,
    ):
        normalized = normalize_method(method)
        if normalized not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported yearly continual method: {method}")
        self.method = normalized
        self.base = StreamingForecastingTrainer(
            model,
            config,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        method_defaults = {
            "memory_per_task": 64,
            "replay_weight": 0.5,
            "distill_weight": 0.1,
            "teacher_ema_decay": 0.99,
            "gate_supervision_weight": 0.1,
            "lwf_lambda": 0.1,
            "ewc_lambda": 1.0,
            "ewc_decay": 1.0,
            "fisher_max_batches": 16,
            "derpp_alpha": 0.1,
        }
        for key, value in method_defaults.items():
            self.base.config.setdefault(key, value)
        self.model = self.base.model
        self.config = self.base.config
        self.device = self.base.device
        self.output_dir = self.base.output_dir
        self.checkpoint_dir = self.base.checkpoint_dir
        self.memory = StreamingReplayMemory(
            capacity_per_task=int(self.config.get("memory_per_task", 64)),
            seed=int(self.config.get("seed", 42)),
        )
        self.ewc = OnlineEWCState()
        self.previous_model: nn.Module | None = None
        self.ema_teacher: nn.Module | None = None
        self.teacher_gate = TeacherGate().to(self.device)
        self.last_retention_metrics: dict[str, Any] = {}
        self.standardizer_bank: dict[str, Any] = {}
        self.active_standardizer_dataset: str | None = None
        self.dataset_loss_audit: dict[str, dict[str, Any]] = {}

    def _use_memory(self) -> bool:
        return self.method in {"replay", "derpp", "expert_replay_distill"}

    def _use_balanced_replay(self) -> bool:
        return self.method == "expert_replay_distill"

    def _use_ema_teacher(self) -> bool:
        return self.method == "expert_replay_distill" and (
            float(self.config.get("distill_weight", 0.1)) > 0.0
            or float(self.config.get("gate_supervision_weight", 0.1)) > 0.0
        )

    def _use_historical_anchor(self) -> bool:
        return self._use_ema_teacher()

    def _use_teacher_gate(self) -> bool:
        return self._use_ema_teacher() and float(self.config.get("gate_supervision_weight", 0.1)) > 0.0

    def _use_teacher_distillation(self) -> bool:
        return self._use_ema_teacher() and float(self.config.get("distill_weight", 0.1)) > 0.0

    @property
    def standardizer(self):
        return self.base.standardizer

    @standardizer.setter
    def standardizer(self, value) -> None:
        self.base.standardizer = value
        if value is not None:
            self.base._activate_standardizer_loss()

    def _dataset_name(
            self,
            dataset: Dataset,
            *,
            task_name: str | None = None,
    ) -> str:
        """
        Recover the physical dataset identity of one continual task.

        Domain IDs are global, but normalization statistics must remain
        dataset-specific.
        """
        period = getattr(dataset, "period", None)

        if (
                period is not None
                and getattr(period, "dataset_name", None)
        ):
            return str(period.dataset_name)

        for attribute in (
                "dataset_name",
                "source_dataset",
        ):
            value = getattr(dataset, attribute, None)

            if value is not None:
                return str(value)

        if task_name is not None:
            known_datasets = (
                "caltrans_bicycle",
                "caltrans_pedestrian",
                "dc_nova",
                "pvcgn_hangzhou",
                "pvcgn_shanghai",
                "shanghai_metroflow",
                "minneapolis_i94",
                "mta_subway",
                "pems03",
            )

            for dataset_name in known_datasets:
                if str(task_name).startswith(
                        dataset_name + "_"
                ):
                    return dataset_name

        raise ValueError(
            "cannot determine physical dataset name for "
            f"task={task_name!r}, dataset={type(dataset).__name__}"
        )

    def _fit_dataset_standardizer(
            self,
            dataset_name: str,
            train_dataset: Dataset,
    ) -> None:
        from trainers.streaming_forecasting_trainer import (
            StreamingStandardizer,
        )

        if dataset_name in self.standardizer_bank:
            return

        print(
            f"[Standardizer Bank] FIT dataset={dataset_name} "
            f"from first task only",
            flush=True,
        )

        standardizer = StreamingStandardizer.fit(
            train_dataset,
            target_mode=self.base.target_mode,
            flow_feature_index=self.base.flow_feature_index,
            max_windows=int(
                self.config.get(
                    "standardizer_max_windows",
                    2048,
                )
            ),
        )

        self.standardizer_bank[dataset_name] = standardizer

    def _activate_standardizer_by_name(
            self,
            dataset_name: str,
    ) -> None:
        if dataset_name not in self.standardizer_bank:
            raise KeyError(
                "standardizer is unavailable for dataset "
                f"{dataset_name!r}"
            )

        self.base.standardizer = (
            self.standardizer_bank[dataset_name]
        )
        loss_audit = self.base._activate_standardizer_loss()
        if self.dataset_loss_audit.get(dataset_name) != loss_audit:
            self.dataset_loss_audit[dataset_name] = loss_audit
            if loss_audit["adaptive_loss"]:
                print(f"[TRAIN Loss Policy] dataset={dataset_name} loss={loss_audit['loss']} "
                      f"target_zero_fraction={loss_audit['target_zero_fraction']:.6f} "
                      f"valid_targets={loss_audit['target_valid_count']} "
                      "source=first_task_train_standardizer_samples", flush=True)
        self.active_standardizer_dataset = dataset_name

    def _activate_standardizer(
            self,
            dataset: Dataset,
            *,
            task_name: str | None = None,
    ) -> str:
        dataset_name = self._dataset_name(
            dataset,
            task_name=task_name,
        )

        self._activate_standardizer_by_name(
            dataset_name
        )

        return dataset_name

    def fit_task(
        self,
        task_name: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        *,
        refit_standardizer: bool,
        checkpoint_name: str,
    ) -> dict[str, Any]:
        dataset_name = self._dataset_name(
            train_dataset,
            task_name=task_name,
        )

        if dataset_name not in self.standardizer_bank:
            self._fit_dataset_standardizer(
                dataset_name,
                train_dataset,
            )

        self._activate_standardizer_by_name(
            dataset_name
        )

        if refit_standardizer:
            print(
                f"[Standardizer Bank] request refit=True "
                f"dataset={dataset_name}; "
                f"existing bank entry is preserved",
                flush=True,
            )

        self._configure_model(self.model, train_dataset)
        train_loader = self.base._loader(train_dataset, shuffle=True)
        val_loader = self.base._loader(val_dataset, shuffle=False)
        train_adj = self.base._adjacency_tensor(train_dataset)
        val_adj = self.base._adjacency_tensor(val_dataset)

        epochs = int(self.config.get("epochs", 5))
        patience = int(self.config.get("patience", 0))
        min_delta = float(self.config.get("min_delta", 0.0))
        best_val = float("inf")
        best_path = self.checkpoint_dir / checkpoint_name
        last_path = best_path.with_name(f"{best_path.stem}.last.pt")
        history = []
        epochs_without_improvement = 0
        start_epoch = 1
        last_state = try_load_torch(last_path, map_location=self.device)
        if last_state is not None and last_state.get("task_name") == task_name:
            self.model.load_state_dict(last_state["model_state"])
            if last_state.get("standardizer_bank"):
                self.standardizer_bank = dict(
                    last_state["standardizer_bank"]
                )

                self.active_standardizer_dataset = (
                    last_state.get(
                        "active_standardizer_dataset",
                        dataset_name,
                    )
                )

                self._activate_standardizer_by_name(
                    dataset_name
                )

            else:
                # Backward compatibility with old single-dataset checkpoints.
                self.base.standardizer = (
                    last_state["standardizer"]
                )

                self.standardizer_bank[
                    dataset_name
                ] = self.base.standardizer

                self.active_standardizer_dataset = (
                    dataset_name
                )
            history = list(last_state["history"])
            if last_state.get("continual_state") is not None:
                replay_datasets = {entry.name: entry.dataset for entry in self.memory.entries}
                self.load_continual_state(last_state["continual_state"], replay_datasets)
            best_val = float(last_state["best_val_loss"])
            epochs_without_improvement = int(last_state["epochs_without_improvement"])
            start_epoch = int(last_state["next_epoch"])
            print(
                f"Resume task={task_name} from epoch={start_epoch}/{epochs}",
                flush=True,
            )

        self._activate_standardizer_by_name(dataset_name)
        self._configure_model(self.model, train_dataset)
        if last_state is not None and getattr(self.model, "active_domain_id", None) is not None:
            self.base._remember_routing_domain(
                self.model,
                (train_dataset, val_dataset),
                int(self.model.active_domain_id),
                task_id=task_name,
            )
        else:
            decision = self.base.prepare_routing_for_dataset(
                train_dataset,
                task_id=task_name,
                allow_create=True,
                update_prototype=True,
                record_assignment=True,
            )
            if decision is not None:
                self.base._remember_routing_domain(
                    self.model,
                    (val_dataset,),
                    int(decision.selected_id),
                    task_id=task_name,
                )

        if self.previous_model is not None and self.method == "lwf":
            previous_decision = self.base.prepare_routing_for_dataset(
                train_dataset,
                task_id=task_name,
                allow_create=False,
                update_prototype=False,
                record_assignment=False,
                model=self.previous_model,
            )
            if previous_decision is not None:
                self.base._remember_routing_domain(
                    self.previous_model,
                    (val_dataset,),
                    int(previous_decision.selected_id),
                    task_id=task_name,
                )

        trainable_parameters = list(self.model.parameters())
        if self._use_teacher_gate():
            trainable_parameters.extend(self.teacher_gate.parameters())
        if self._use_ema_teacher():
            self._ensure_ema_teacher()
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(self.config.get("lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        if last_state is not None and last_state.get("task_name") == task_name:
            if last_state.get("optimizer_state_by_name") is not None:
                self.base._load_optimizer_state_by_name(
                    optimizer,
                    self.model,
                    last_state["optimizer_state_by_name"],
                    extra_parameters={"teacher_gate." + n: p for n, p in self.teacher_gate.named_parameters()},
                )
            else:
                optimizer.load_state_dict(last_state["optimizer_state"])

        if last_state is not None:
            restore_rng_state(last_state.get("rng_state"))
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self._train_epoch(
                train_dataset,
                train_loader,
                train_adj,
                optimizer,
                progress_label=f"TRAIN epoch={epoch:03d}/{epochs:03d} task={task_name}",
            )
            self._configure_model(self.model, val_dataset)
            val_loss = self.base._loss_epoch(
                val_loader,
                val_adj,
                progress_label=f"VAL epoch={epoch:03d}/{epochs:03d} task={task_name}",
            )
            self._configure_model(self.model, train_dataset)
            record = {"epoch": epoch, **train_metrics, "val_loss": val_loss}
            history.append(record)
            if val_loss < best_val - min_delta:
                best_val = val_loss
                epochs_without_improvement = 0
                atomic_torch_save(
                    {
                        "model_state": self.model.state_dict(),
                        "standardizer": self.base.standardizer,
                        "config": self.config,
                        "standardizer_bank": dict(
                            self.standardizer_bank
                        ),
                        "active_standardizer_dataset":
                            self.active_standardizer_dataset,
                        # The selected student, EMA teacher, gate and replay
                        # sampler must come from one training trajectory.
                        "continual_state": self.continual_state_dict(),
                        "rng_state": capture_rng_state(),
                        "method": self.method,
                        "history": history,
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1
            print(
                f"epoch={epoch:03d} train_loss={record['train_loss']:.6f} "
                f"replay_loss={record['replay_loss']:.6f} "
                f"regularizer={record['regularizer_loss']:.6f} val_loss={val_loss:.6f}"
            )
            atomic_torch_save(
                {
                    "task_name": task_name,
                    "next_epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "optimizer_state_by_name": self.base._optimizer_state_by_name(
                        optimizer, self.model,
                        extra_parameters={"teacher_gate." + n: p for n, p in self.teacher_gate.named_parameters()},
                    ),
                    "continual_state": self.continual_state_dict(),
                    "rng_state": capture_rng_state(),
                    "standardizer": self.base.standardizer,
                    "standardizer_bank": dict(
                        self.standardizer_bank
                    ),

                    "active_standardizer_dataset":
                        self.active_standardizer_dataset,
                    "history": history,
                    "best_val_loss": best_val,
                    "epochs_without_improvement": epochs_without_improvement,
                    "method": self.method,
                },
                last_path,
            )
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"early_stop epoch={epoch:03d} best_val_loss={best_val:.6f}")
                break

        if best_path.exists():
            state = try_load_torch(best_path, map_location=self.device)
            if state is None:
                raise OSError(f"best checkpoint is unreadable after training: {best_path}")
            self.model.load_state_dict(state["model_state"])
            if state.get("continual_state") is not None:
                # Capture dataset objects before load_continual_state rebuilds
                # the replay entries. The current task is added only below.
                replay_datasets = {entry.name: entry.dataset for entry in self.memory.entries}
                self.load_continual_state(state["continual_state"], replay_datasets)
                self._activate_standardizer_by_name(dataset_name)
            # Loading dynamic modules can consume RNG, so restore it last.
            restore_rng_state(state.get("rng_state"))
        if getattr(self.model, "active_domain_id", None) is not None:
            self.base._remember_routing_domain(
                self.model,
                (train_dataset, val_dataset),
                int(self.model.active_domain_id),
                task_id=task_name,
            )
        self._configure_model(self.model, train_dataset)

        if self.method == "ewc":
            fisher = self._estimate_fisher(train_dataset)
            self.ewc.consolidate(
                self.model,
                fisher,
                decay=float(self.config.get("ewc_decay", 1.0)),
            )

        if self._use_memory():
            entry = self.memory.add(
                task_name,
                train_dataset,
                dataset_name=dataset_name,
            )
            if entry is not None:
                self._annotate_replay_entry(entry)

                if self._use_historical_anchor():
                    entry.anchor_pred = self._anchor_predictions(entry)

                if self.method == "derpp":
                    entry.logits = self._stored_logits(entry)

        # Refresh Domain prototypes in the current backbone latent space
        # after the best checkpoint and replay-memory metadata are fixed.
        if self.method == "expert_replay_distill":
            self._refresh_domain_prototypes_from_memory()
            self._refresh_state_prototypes_from_memory()

        if self._use_ema_teacher():
            self._ensure_ema_teacher()
        else:
            if self.method == "lwf":
                self.previous_model = copy.deepcopy(
                    self.model
                ).to(self.device).eval()

                for parameter in self.previous_model.parameters():
                    parameter.requires_grad = False
            else:
                self.previous_model = None

        routing = self.base._routing_summary(self.model)
        if routing is not None:
            print(f"[State Routing] {routing['state_routing']}", flush=True)
        return {
            "history": history,
            "best_val_loss": best_val,
            "checkpoint": str(best_path),
            "memory_size": self.memory.size,
            "retention": self.last_retention_metrics,
            "routing": routing,
        }

    def evaluate(
            self,
            dataset: Dataset,
            *,
            task_name: str | None = None,
            domain_id: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate one task without changing the persistent TraffiCL routing state.

        For historical-task evaluation:
          - restore the dataset-specific standardizer;
          - force the Domain Expert recorded when that task was learned;
          - keep State routing adaptive among the existing State Experts;
          - disable routing updates so evaluation cannot create/update experts;
          - restore the model/scaler state that was active before evaluation.
        """

        original_standardizer_dataset = (
            self.active_standardizer_dataset
        )

        original_domain_id = getattr(
            self.model,
            "active_domain_id",
            None,
        )

        original_task_id = getattr(
            self.model,
            "active_task_id",
            None,
        )

        routing_updates = getattr(
            self.model,
            "routing_updates_enabled",
            None,
        )

        try:
            # Use the scaler belonging to the physical dataset
            # of the task being evaluated.
            self._activate_standardizer(
                dataset
            )

            # Historical evaluation must reuse the Domain that
            # was assigned when this task was originally learned.
            if domain_id is not None:
                self.base._remember_routing_domain(
                    self.model,
                    (dataset,),
                    int(domain_id),
                    task_id=task_name,
                )

                if hasattr(
                        self.model,
                        "activate_recorded_domain",
                ):
                    self.model.activate_recorded_domain(
                        int(domain_id),
                        task_id=task_name,
                    )

            # Evaluation must not create new Domain/State experts
            # or update prototypes.
            #
            # State routing itself is still active, so each test
            # window can adaptively select among EXISTING State
            # Experts inside the recorded Domain.
            if hasattr(
                    self.model,
                    "set_routing_updates",
            ):
                self.model.set_routing_updates(
                    False
                )

            return self.base.evaluate(
                dataset
            )

        finally:
            # Restore routing-update mode.
            if (
                    routing_updates is not None
                    and hasattr(
                self.model,
                "set_routing_updates",
            )
            ):
                self.model.set_routing_updates(
                    bool(routing_updates)
                )

            # Restore the task/domain that was active before the
            # historical evaluation.
            if (
                    original_domain_id is not None
                    and hasattr(
                self.model,
                "activate_recorded_domain",
            )
            ):
                self.model.activate_recorded_domain(
                    int(original_domain_id),
                    task_id=original_task_id,
                )

            # Restore the scaler that was active before evaluation.
            if (
                    original_standardizer_dataset
                    is not None
            ):
                self._activate_standardizer_by_name(
                    original_standardizer_dataset
                )

    def continual_state_dict(self) -> dict[str, Any]:
        return {
            "standardizer_bank": dict(
                self.standardizer_bank
            ),
            "dataset_loss_audit": dict(getattr(self, "dataset_loss_audit", {})),

            "active_standardizer_dataset":
                self.active_standardizer_dataset,
            "memory": [
                {
                    "name": entry.name,
                    "dataset_name": entry.dataset_name,
                    "indices": entry.indices,
                    "logits": entry.logits,
                    "anchor_pred": entry.anchor_pred,
                    "domain_id": entry.domain_id,
                    "state_ids": entry.state_ids,
                }
                for entry in self.memory.entries
            ],
            "memory_rng_state": self.memory.rng.bit_generator.state,
            "balanced_replay_cursor": self.memory.balanced_replay_cursor,
            "ema_teacher_state": (
                None if self.ema_teacher is None else self.ema_teacher.state_dict()
            ),
            "teacher_gate_state": self.teacher_gate.state_dict(),
            "teacher_ema_decay": float(self.config.get("teacher_ema_decay", 0.99)),
            "gate_supervision_weight": float(self.config.get("gate_supervision_weight", 0.1)),
            "ewc_fisher": {
                name: value.detach().cpu() for name, value in self.ewc.fisher.items()
            },
            "ewc_means": {
                name: value.detach().cpu() for name, value in self.ewc.means.items()
            },
        }

    def load_continual_state(
        self,
        state: Mapping[str, Any],
        datasets_by_name: Mapping[str, Dataset],
    ) -> None:
        self.standardizer_bank = dict(
            state.get(
                "standardizer_bank",
                {},
            )
        )
        # Every activation verifies this audit against restored TRAIN statistics.
        self.dataset_loss_audit = copy.deepcopy(state.get("dataset_loss_audit", {}))

        self.active_standardizer_dataset = (
            state.get(
                "active_standardizer_dataset"
            )
        )

        if (
                self.active_standardizer_dataset
                is not None
                and self.active_standardizer_dataset
                in self.standardizer_bank
        ):
            self._activate_standardizer_by_name(
                self.active_standardizer_dataset
            )
        self.memory.entries = []
        for item in state.get("memory", []):
            name = str(item["name"])
            if name not in datasets_by_name:
                raise KeyError(f"resume memory task is unavailable: {name}")
            self.memory.entries.append(
                ReplayEntry(
                    name=name,
                    dataset=datasets_by_name[name],
                    indices=np.asarray(item["indices"], dtype=np.int64),
                    logits=(
                        None
                        if item.get("logits") is None
                        else np.asarray(item["logits"], dtype=np.float32)
                    ),
                    anchor_pred=(
                        None
                        if item.get("anchor_pred") is None
                        else np.asarray(item["anchor_pred"], dtype=np.float32)
                    ),
                    domain_id=(
                        None if item.get("domain_id") is None else int(item["domain_id"])
                    ),
                    state_ids=(
                        None
                        if item.get("state_ids") is None
                        else np.asarray(item["state_ids"], dtype=np.int64)
                    ),
                    dataset_name=(
                        None
                        if item.get("dataset_name") is None
                        else str(item["dataset_name"])
                    )
                )
            )
        if state.get("memory_rng_state") is not None:
            self.memory.rng.bit_generator.state = state["memory_rng_state"]
        self.memory.balanced_replay_cursor = int(state.get("balanced_replay_cursor", 0))
        self.ewc.fisher = {
            name: value.to(self.device)
            for name, value in state.get("ewc_fisher", {}).items()
        }
        self.ewc.means = {
            name: value.to(self.device)
            for name, value in state.get("ewc_means", {}).items()
        }
        self.previous_model = copy.deepcopy(self.model).to(self.device).eval()
        for parameter in self.previous_model.parameters():
            parameter.requires_grad = False
        if state.get("teacher_gate_state") is not None:
            self.teacher_gate.load_state_dict(state["teacher_gate_state"])
        self.teacher_gate.to(self.device)
        if self._use_ema_teacher() and state.get("ema_teacher_state") is None:
            self._ensure_ema_teacher()
        elif self._use_ema_teacher():
            self._ensure_ema_teacher()
            assert self.ema_teacher is not None
            self.ema_teacher.load_state_dict(
                state["ema_teacher_state"],
                strict=True,
            )
            self._freeze_teacher(self.ema_teacher)

    def _train_epoch(
        self,
        train_dataset: Dataset,
        loader,
        adjacency: torch.Tensor | None,
        optimizer: torch.optim.Optimizer,
        *,
        progress_label: str | None = None,
    ) -> dict[str, float]:
        self._configure_model(self.model, train_dataset)
        self.model.train()
        totals = {"train": 0.0, "replay": 0.0, "regularizer": 0.0}
        batches = 0
        total_batches = self.base._batch_count(loader, "max_train_batches")
        progress = LogProgressBar(progress_label, total_batches) if progress_label else None

        for batch_index, batch in enumerate(
            self.base._limited_batches(loader, "max_train_batches"),
            start=1,
        ):
            x, y, y_mask = unpack_batch(batch)
            x = x.to(self.device).float()
            y = y.to(self.device).float()
            y_mask = y_mask.to(self.device).bool()
            if not bool(y_mask.any()):
                continue

            self._configure_model(self.model, train_dataset)
            optimizer.zero_grad(set_to_none=True)
            pred = self.base._model_forward(self.base._transform_x(x), adjacency)
            self.base._sync_optimizer_parameters(optimizer, self.model)
            target = self.base._transform_y(self.base._target_array(x, y))
            supervised = self.base._masked_loss(pred, target, y_mask)
            regularizer = pred.new_zeros(())

            if self.method == "lwf" and self.previous_model is not None:
                self._configure_model(self.previous_model, train_dataset)
                with torch.no_grad():
                    teacher = self._forward_model(
                        self.previous_model,
                        self.base._transform_x(x),
                        adjacency,
                    )
                regularizer = float(self.config.get("lwf_lambda", 0.1)) * masked_mse(
                    pred,
                    teacher,
                    y_mask,
                )
            elif self.method == "ewc" and self.ewc.fisher:
                regularizer = float(self.config.get("ewc_lambda", 1.0)) * self.ewc.penalty(
                    self.model
                )

            (supervised + regularizer).backward()
            replay_loss = self._backward_replay(optimizer, current_loss=supervised.detach())
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.config.get("grad_clip", 1.0)),
            )
            optimizer.step()
            if self._use_ema_teacher():
                self._update_ema_teacher()
            self._configure_model(self.model, train_dataset)

            totals["train"] += float(supervised.detach().item())
            totals["replay"] += replay_loss
            totals["regularizer"] += float(regularizer.detach().item())
            batches += 1
            if progress is not None:
                progress.update(
                    batch_index,
                    loss=totals["train"] / max(batches, 1),
                    replay=totals["replay"] / max(batches, 1),
                    regularizer=totals["regularizer"] / max(batches, 1),
                )

        divisor = max(batches, 1)
        return {
            "train_loss": totals["train"] / divisor,
            "replay_loss": totals["replay"] / divisor,
            "regularizer_loss": totals["regularizer"] / divisor,
        }

    def _backward_replay(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        current_loss: torch.Tensor | None = None,
    ) -> float:
        current_standardizer_dataset = (
            self.active_standardizer_dataset
        )
        if self.method not in {"replay", "derpp", "expert_replay_distill"}:
            return 0.0
        if self._use_balanced_replay():
            return self._backward_balanced_expert_replay(
                optimizer,
                current_loss=current_loss,
            )
        sampled = self.memory.sample(int(self.config.get("batch_size", 64)))
        if sampled is None:
            return 0.0

        entry, x_np, y_np, mask_np, stored_logits = sampled

        replay_dataset_name = (
            entry.dataset_name
            if entry.dataset_name is not None
            else self._dataset_name(
                entry.dataset,
                task_name=entry.name,
            )
        )

        # Replay must use the normalization context of the
        # historical physical dataset.
        self._activate_standardizer_by_name(
            replay_dataset_name
        )

        x = torch.from_numpy(
            x_np
        ).to(self.device)

        y = torch.from_numpy(
            y_np
        ).to(self.device)

        y_mask = torch.from_numpy(
            mask_np
        ).to(self.device).bool()
        if entry.domain_id is not None and hasattr(self.model, "activate_recorded_domain"):
            self.base._remember_routing_domain(
                self.model,
                (entry.dataset,),
                entry.domain_id,
                task_id=entry.name,
            )
        elif self.base.activate_routing_for_dataset(entry.dataset) is None:
            self.base.prepare_routing_for_dataset(
                entry.dataset,
                task_id=entry.name,
                allow_create=False,
                update_prototype=False,
                record_assignment=False,
            )
        self._configure_model(self.model, entry.dataset)
        adjacency = self.base._adjacency_tensor(entry.dataset)
        routing_updates = getattr(self.model, "routing_updates_enabled", None)
        if hasattr(self.model, "set_routing_updates"):
            self.model.set_routing_updates(False)
        try:
            pred = self.base._model_forward(self.base._transform_x(x), adjacency)
        finally:
            if routing_updates is not None:
                self.model.set_routing_updates(bool(routing_updates))
        self.base._sync_optimizer_parameters(optimizer, self.model)
        target = self.base._transform_y(self.base._target_array(x, y))
        replay = float(self.config.get("replay_weight", 0.5)) * self.base._masked_loss(
            pred,
            target,
            y_mask,
        )
        extra = pred.new_zeros(())

        if self.method == "expert_replay_distill" and self.previous_model is not None:
            if entry.domain_id is not None and hasattr(
                self.previous_model, "activate_recorded_domain"
            ):
                self.base._remember_routing_domain(
                    self.previous_model,
                    (entry.dataset,),
                    entry.domain_id,
                    task_id=entry.name,
                )
            self._configure_model(self.previous_model, entry.dataset)
            with torch.no_grad():
                teacher = self._forward_model(
                    self.previous_model,
                    self.base._transform_x(x),
                    adjacency,
                )
            extra = float(self.config.get("distill_weight", 0.1)) * masked_mse(
                pred,
                teacher,
                y_mask,
            )
        elif self.method == "derpp" and stored_logits is not None:
            logits = torch.from_numpy(stored_logits).to(self.device).float()
            extra = float(self.config.get("derpp_alpha", 0.1)) * masked_mse(
                pred,
                logits,
                y_mask,
            )

        total = replay + extra
        total.backward()
        if current_standardizer_dataset is not None:
            self._activate_standardizer_by_name(
                current_standardizer_dataset
            )

        return float(
            total.detach().item()
        )

    def _backward_balanced_expert_replay(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        current_loss: torch.Tensor | None,
    ) -> float:
        current_standardizer_dataset = (
            self.active_standardizer_dataset
        )
        groups = self.memory.sample_balanced(int(self.config.get("batch_size", 64)))
        if not groups:
            self.last_retention_metrics = {
                "memory_size_total": self.memory.size,
                "memory_size_by_task": self.memory.size_by_task(),
                "memory_size_by_dataset":
                    self.memory.size_by_dataset(),
                "memory_task_count_by_dataset":
                    self.memory.task_count_by_dataset(),
                "historical_anchor_available_fraction": 0.0,
            }
            return 0.0
        use_teacher = self._use_ema_teacher()
        if use_teacher:
            self._ensure_ema_teacher()
            assert self.ema_teacher is not None
        total_samples = sum(group.sample_count for group in groups)
        replay_supervised = None
        teacher_distill = None
        teacher_gate = None
        alpha_values: list[torch.Tensor] = []
        drift_values: list[float] = []
        quality_values: list[float] = []
        anchor_available = 0
        replay_samples_by_task: dict[str, int] = {}

        for group in groups:
            replay_samples_by_task[group.task_name] = (
                replay_samples_by_task.get(group.task_name, 0) + group.sample_count
            )
            weight = float(group.sample_count) / float(max(total_samples, 1))
            pred, target, y_mask, adjacency = self._forward_replay_group_student(
                optimizer,
                group,
            )
            group_replay = self.base._masked_loss(pred, target, y_mask)
            replay_supervised = (
                weight * group_replay
                if replay_supervised is None
                else replay_supervised + weight * group_replay
            )
            if use_teacher:
                with torch.no_grad():
                    ema_pred = self._forward_replay_group_teacher(group, adjacency)
                if group.anchor_pred is None:
                    anchor_pred = ema_pred.detach()
                else:
                    anchor_pred = torch.from_numpy(group.anchor_pred).to(self.device).float()
                    anchor_available += group.sample_count
                drift = masked_mae(ema_pred.detach(), anchor_pred.detach(), y_mask)
                quality = torch.exp(-masked_mae(anchor_pred.detach(), target.detach(), y_mask))
                pressure = current_loss
                if pressure is None:
                    pressure = pred.detach().new_zeros(())
                if self._use_teacher_gate():
                    gate_features = torch.stack(
                        [
                            torch.nan_to_num(drift.detach(), nan=0.0, posinf=1e4, neginf=0.0).clamp(0.0, 1e4),
                            torch.nan_to_num(quality.detach(), nan=0.0, posinf=1e4, neginf=0.0).clamp(0.0, 1e4),
                            torch.nan_to_num(pressure.detach(), nan=0.0, posinf=1e4, neginf=0.0).clamp(0.0, 1e4),
                        ]
                    ).view(1, 3)
                    alpha = self.teacher_gate(gate_features.to(self.device)).view(())
                else:
                    alpha = pred.detach().new_zeros(())
                teacher_target_for_gate = (1.0 - alpha) * ema_pred.detach() + alpha * anchor_pred.detach()
                if self._use_teacher_gate():
                    group_gate = self.base._masked_loss(teacher_target_for_gate, target.detach(), y_mask)
                    teacher_gate = (
                        weight * group_gate if teacher_gate is None else teacher_gate + weight * group_gate
                    )
                    alpha_values.append(alpha.detach().view(1))
                if self._use_teacher_distillation():
                    group_distill = self._teacher_distillation_loss(
                        pred, teacher_target_for_gate.detach(), y_mask)
                    teacher_distill = (
                        weight * group_distill
                        if teacher_distill is None
                        else teacher_distill + weight * group_distill
                    )
                drift_values.append(float(drift.detach().cpu()))
                quality_values.append(float(quality.detach().cpu()))

        zero = next(self.model.parameters()).new_zeros(())
        replay_supervised = zero if replay_supervised is None else replay_supervised
        teacher_distill = zero if teacher_distill is None else teacher_distill
        teacher_gate = zero if teacher_gate is None else teacher_gate
        replay_weight = float(self.config.get("replay_weight", 0.5))
        distill_weight = float(self.config.get("distill_weight", 0.1))
        gate_weight = float(self.config.get("gate_supervision_weight", 0.1))
        total = replay_weight * replay_supervised + distill_weight * teacher_distill + gate_weight * teacher_gate
        total.backward()

        if current_standardizer_dataset is not None:
            self._activate_standardizer_by_name(
                current_standardizer_dataset
            )

        counts = list(replay_samples_by_task.values())
        alpha_tensor = (
            torch.cat(alpha_values).float().cpu()
            if alpha_values
            else torch.zeros(0, dtype=torch.float32)
        )
        self.last_retention_metrics = {
            "memory_size_total": self.memory.size,
            "memory_size_by_task": self.memory.size_by_task(),
            "memory_size_by_dataset":
                self.memory.size_by_dataset(),

            "memory_task_count_by_dataset":
                self.memory.task_count_by_dataset(),
            "replay_samples_by_task": replay_samples_by_task,
            "replay_fraction_by_task": {
                key: float(value) / float(max(total_samples, 1))
                for key, value in replay_samples_by_task.items()
            },
            "replay_balance_max_min_gap": int(max(counts) - min(counts)) if counts else 0,
            "replay_supervised_loss": float(replay_supervised.detach().cpu()),
            "teacher_distill_loss": float(teacher_distill.detach().cpu()),
            "teacher_gate_loss": float(teacher_gate.detach().cpu()),
            "teacher_alpha_mean": float(alpha_tensor.mean()) if alpha_tensor.numel() else 0.0,
            "teacher_alpha_std": float(alpha_tensor.std(unbiased=False)) if alpha_tensor.numel() else 0.0,
            "teacher_alpha_min": float(alpha_tensor.min()) if alpha_tensor.numel() else 0.0,
            "teacher_alpha_max": float(alpha_tensor.max()) if alpha_tensor.numel() else 0.0,
            "ema_anchor_drift": float(np.mean(drift_values)) if drift_values else 0.0,
            "anchor_quality": float(np.mean(quality_values)) if quality_values else 0.0,
            "ema_teacher_student_parameter_delta": self._ema_student_delta(),
            "historical_anchor_available_fraction": float(anchor_available) / float(max(total_samples, 1)),
        }
        return float(total.detach().item())

    def _teacher_distillation_loss(self, prediction, teacher_target, mask):
        """Use a bounded-gradient teacher objective when explicitly requested.

        The active objective belongs to the replay sample's physical dataset;
        never infer it from the current task or from held-out errors.
        """
        objective = str(self.config.get("teacher_distillation_loss", "mse"))
        if objective == "supervised":
            return self.base._masked_loss(prediction, teacher_target.detach(), mask)
        if objective == "mse":
            return masked_mse(prediction, teacher_target.detach(), mask)
        raise ValueError(f"unsupported teacher_distillation_loss: {objective}")

    def _forward_replay_group_student(
            self,
            optimizer: torch.optim.Optimizer,
            group: ReplayGroup,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = torch.from_numpy(group.x).to(self.device)
        if group.dataset_name is None:
            replay_dataset_name = self._dataset_name(
                group.entry.dataset,
                task_name=group.task_name,
            )
        else:
            replay_dataset_name = group.dataset_name

        self._activate_standardizer_by_name(
            replay_dataset_name
        )
        y = torch.from_numpy(group.y).to(self.device)
        y_mask = torch.from_numpy(group.mask).to(self.device).bool()

        # Restore the historical Domain context first.
        self._activate_replay_context(self.model, group)
        self._configure_model(self.model, group.entry.dataset)
        adjacency = self.base._adjacency_tensor(group.entry.dataset)

        routing_updates = getattr(
            self.model,
            "routing_updates_enabled",
            None,
        )

        # Preserve any pre-existing forced-state context.
        previous_forced_state_ids = getattr(
            self.model,
            "_forced_state_ids",
            None,
        )

        if hasattr(self.model, "set_routing_updates"):
            self.model.set_routing_updates(False)

        # Replay must use the historical State Expert assignment,
        # rather than re-routing with the current state prototypes.
        if hasattr(self.model, "set_forced_state_ids"):
            self.model.set_forced_state_ids(
                None
                if group.state_ids is None
                else group.state_ids.tolist()
            )

        try:
            pred = self.base._model_forward(
                self.base._transform_x(x),
                adjacency,
            )
        finally:
            # Never allow replay routing context to leak into
            # the current-task training batch.
            if hasattr(self.model, "set_forced_state_ids"):
                self.model.set_forced_state_ids(
                    previous_forced_state_ids
                )

            if routing_updates is not None:
                self.model.set_routing_updates(
                    bool(routing_updates)
                )

        # Dynamic experts/gates may have been created earlier;
        # keep optimizer synchronization unchanged.
        self.base._sync_optimizer_parameters(
            optimizer,
            self.model,
        )

        target = self.base._transform_y(
            self.base._target_array(x, y)
        )

        return pred, target, y_mask, adjacency

    def _forward_replay_group_teacher(
            self,
            group: ReplayGroup,
            adjacency: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.ema_teacher is not None

        x = torch.from_numpy(group.x).to(self.device)
        if group.dataset_name is None:
            replay_dataset_name = self._dataset_name(
                group.entry.dataset,
                task_name=group.task_name,
            )
        else:
            replay_dataset_name = (
                group.dataset_name
            )

        self._activate_standardizer_by_name(
            replay_dataset_name
        )
        # Restore the same historical Domain context for the teacher.
        self._activate_replay_context(
            self.ema_teacher,
            group,
        )
        self._configure_model(
            self.ema_teacher,
            group.entry.dataset,
        )

        routing_updates = getattr(
            self.ema_teacher,
            "routing_updates_enabled",
            None,
        )

        previous_forced_state_ids = getattr(
            self.ema_teacher,
            "_forced_state_ids",
            None,
        )

        if hasattr(
                self.ema_teacher,
                "set_routing_updates",
        ):
            self.ema_teacher.set_routing_updates(False)

        # Student and EMA teacher must evaluate the replay sample
        # under the SAME historical State Expert assignment.
        if hasattr(
                self.ema_teacher,
                "set_forced_state_ids",
        ):
            self.ema_teacher.set_forced_state_ids(
                None
                if group.state_ids is None
                else group.state_ids.tolist()
            )

        try:
            return self._forward_model(
                self.ema_teacher,
                self.base._transform_x(x),
                adjacency,
            )
        finally:
            if hasattr(
                    self.ema_teacher,
                    "set_forced_state_ids",
            ):
                self.ema_teacher.set_forced_state_ids(
                    previous_forced_state_ids
                )

            if routing_updates is not None:
                self.ema_teacher.set_routing_updates(
                    bool(routing_updates)
                )

    def _activate_replay_context(self, model: nn.Module, group: ReplayGroup) -> None:
        if group.domain_id is not None and hasattr(model, "activate_recorded_domain"):
            self.base._remember_routing_domain(
                model,
                (group.entry.dataset,),
                group.domain_id,
                task_id=group.task_name,
            )
        elif self.base.activate_routing_for_dataset(group.entry.dataset, model=model) is None:
            self.base.prepare_routing_for_dataset(
                group.entry.dataset,
                task_id=group.task_name,
                allow_create=False,
                update_prototype=False,
                record_assignment=False,
                model=model,
            )

    def _ensure_ema_teacher(self) -> None:
        if self.ema_teacher is None:
            self.ema_teacher = copy.deepcopy(self.model).to(self.device).eval()
            self._freeze_teacher(self.ema_teacher)
            return
        old_state = {
            key: self._copy_state_value(value)
            for key, value in self.ema_teacher.state_dict().items()
        }
        student_state = self.model.state_dict()
        mismatch = any(
            key not in old_state or not self._state_values_compatible(old_state[key], value)
            for key, value in student_state.items()
        ) or any(key not in student_state for key in old_state)
        if not mismatch:
            return
        new_teacher = copy.deepcopy(self.model).to(self.device).eval()
        new_state = new_teacher.state_dict()
        for key, value in list(new_state.items()):
            if key in old_state and torch.is_tensor(value) and torch.is_tensor(old_state[key]) and old_state[key].shape == value.shape:
                new_state[key].copy_(old_state[key].to(device=value.device, dtype=value.dtype))
        new_teacher.load_state_dict(new_state, strict=True)
        self.ema_teacher = new_teacher
        self._freeze_teacher(self.ema_teacher)

    def _update_ema_teacher(self) -> None:
        self._ensure_ema_teacher()
        assert self.ema_teacher is not None
        beta = float(self.config.get("teacher_ema_decay", 0.99))
        student_state = self.model.state_dict()
        teacher_state = self.ema_teacher.state_dict()
        with torch.no_grad():
            for key, student_value in student_state.items():
                if key not in teacher_state:
                    continue
                teacher_value = teacher_state[key]
                if torch.is_tensor(teacher_value) and torch.is_tensor(student_value):
                    if teacher_value.shape != student_value.shape:
                        continue
                    student_value = student_value.to(device=teacher_value.device)
                    if teacher_value.is_floating_point():
                        teacher_value.mul_(beta).add_(student_value.to(dtype=teacher_value.dtype), alpha=1.0 - beta)
                    else:
                        teacher_value.copy_(student_value.to(dtype=teacher_value.dtype))
                else:
                    teacher_state[key] = copy.deepcopy(student_value)
            self.ema_teacher.load_state_dict(
                teacher_state,
                strict=True,
            )
        self._freeze_teacher(self.ema_teacher)

    @staticmethod
    def _freeze_teacher(model: nn.Module) -> None:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _copy_state_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().clone()
        return copy.deepcopy(value)

    @staticmethod
    def _state_values_compatible(left: Any, right: Any) -> bool:
        if torch.is_tensor(left) or torch.is_tensor(right):
            return torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape
        return type(left) is type(right)

    def _ema_student_delta(self) -> float:
        if self.ema_teacher is None:
            return 0.0
        total = 0.0
        count = 0
        teacher_state = self.ema_teacher.state_dict()
        for key, value in self.model.state_dict().items():
            if key not in teacher_state or not torch.is_tensor(value) or not torch.is_tensor(teacher_state[key]):
                continue
            teacher_value = teacher_state[key].to(device=value.device)
            if not value.is_floating_point() or teacher_value.shape != value.shape:
                continue
            total += float((teacher_value - value).abs().mean().detach().cpu())
            count += 1
        return total / max(count, 1)

    def _estimate_fisher(self, dataset: Dataset) -> dict[str, torch.Tensor]:
        self._configure_model(self.model, dataset)
        loader = self.base._loader(dataset, shuffle=False)
        adjacency = self.base._adjacency_tensor(dataset)
        fisher = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        batches = 0
        self.model.eval()
        limit = int(self.config.get("fisher_max_batches", 16))
        for batch_index, batch in enumerate(loader):
            if batch_index >= limit:
                break
            x, y, y_mask = unpack_batch(batch)
            x = x.to(self.device).float()
            y = y.to(self.device).float()
            y_mask = y_mask.to(self.device).bool()
            if not bool(y_mask.any()):
                continue
            self.model.zero_grad(set_to_none=True)
            pred = self.base._model_forward(self.base._transform_x(x), adjacency)
            target = self.base._transform_y(self.base._target_array(x, y))
            self.base._masked_loss(pred, target, y_mask).backward()
            for name, parameter in self.model.named_parameters():
                if name in fisher and parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().square()
            batches += 1
        divisor = max(batches, 1)
        return {name: value / divisor for name, value in fisher.items()}

    def _annotate_replay_entry(self, entry: ReplayEntry) -> None:
        entry_dataset_name = (
            entry.dataset_name
            if entry.dataset_name is not None
            else self._dataset_name(
                entry.dataset,
                task_name=entry.name,
            )
        )

        self._activate_standardizer_by_name(
            entry_dataset_name
        )
        if not hasattr(self.model, "last_routing_trace"):
            return
        domain_id = getattr(self.model, "active_domain_id", None)
        if domain_id is None:
            return
        summary = self.base._routing_summary(self.model)
        if summary is None or not summary.get("num_state_experts"):
            return
        if summary["num_state_experts"][int(domain_id)] == 0:
            return

        entry.domain_id = int(domain_id)
        self.base._remember_routing_domain(
            self.model,
            (entry.dataset,),
            entry.domain_id,
            task_id=entry.name,
        )
        self._configure_model(self.model, entry.dataset)
        adjacency = self.base._adjacency_tensor(entry.dataset)
        state_ids: list[int] = []
        batch_size = int(self.config.get("batch_size", 64))
        was_training = self.model.training
        routing_updates = getattr(self.model, "routing_updates_enabled", None)
        self.model.eval()
        if hasattr(self.model, "set_routing_updates"):
            self.model.set_routing_updates(False)
        try:
            with torch.no_grad():
                for start in range(0, len(entry.indices), batch_size):
                    xs = []
                    for index in entry.indices[start : start + batch_size]:
                        x, _, _ = unpack_sample(entry.dataset[int(index)])
                        xs.append(x)
                    x_tensor = torch.from_numpy(np.stack(xs).astype(np.float32)).to(self.device)
                    self.base._model_forward(self.base._transform_x(x_tensor), adjacency)
                    state_ids.extend(
                        int(value)
                        for value in self.model.last_routing_trace.get("state_ids", [])
                    )
        finally:
            self.model.train(was_training)
            if routing_updates is not None:
                self.model.set_routing_updates(bool(routing_updates))
        if len(state_ids) != len(entry.indices):
            raise RuntimeError("replay state assignments do not align with replay samples")
        entry.state_ids = np.asarray(state_ids, dtype=np.int64)

    def _stored_logits(self, entry: ReplayEntry) -> np.ndarray:
        entry_dataset_name = (
            entry.dataset_name
            if entry.dataset_name is not None
            else self._dataset_name(
                entry.dataset,
                task_name=entry.name,
            )
        )

        self._activate_standardizer_by_name(
            entry_dataset_name
        )
        self._configure_model(self.model, entry.dataset)
        adjacency = self.base._adjacency_tensor(entry.dataset)
        logits = []
        self.model.eval()
        batch_size = int(self.config.get("batch_size", 64))
        with torch.no_grad():
            for start in range(0, len(entry.indices), batch_size):
                xs = []
                for index in entry.indices[start : start + batch_size]:
                    x, _, _ = unpack_sample(entry.dataset[int(index)])
                    xs.append(x)
                x_tensor = torch.from_numpy(np.stack(xs).astype(np.float32)).to(self.device)
                pred = self.base._model_forward(self.base._transform_x(x_tensor), adjacency)
                logits.append(pred.cpu().numpy().astype(np.float32))
        return np.concatenate(logits, axis=0)

    def _anchor_predictions(self, entry: ReplayEntry) -> np.ndarray:
        entry_dataset_name = (
            entry.dataset_name
            if entry.dataset_name is not None
            else self._dataset_name(
                entry.dataset,
                task_name=entry.name,
            )
        )

        self._activate_standardizer_by_name(
            entry_dataset_name
        )
        self._configure_model(self.model, entry.dataset)

        # Restore the historical Domain Expert used by this task.
        if entry.domain_id is not None and hasattr(
                self.model,
                "activate_recorded_domain",
        ):
            self.base._remember_routing_domain(
                self.model,
                (entry.dataset,),
                entry.domain_id,
                task_id=entry.name,
            )

        adjacency = self.base._adjacency_tensor(entry.dataset)
        anchors = []
        batch_size = int(self.config.get("batch_size", 64))

        was_training = self.model.training
        routing_updates = getattr(
            self.model,
            "routing_updates_enabled",
            None,
        )
        previous_forced_state_ids = getattr(
            self.model,
            "_forced_state_ids",
            None,
        )

        self.model.eval()

        if hasattr(self.model, "set_routing_updates"):
            self.model.set_routing_updates(False)

        try:
            with torch.no_grad():
                for start in range(
                        0,
                        len(entry.indices),
                        batch_size,
                ):
                    end = min(
                        start + batch_size,
                        len(entry.indices),
                    )

                    xs = []

                    for index in entry.indices[start:end]:
                        x, _, _ = unpack_sample(
                            entry.dataset[int(index)]
                        )
                        xs.append(x)

                    x_tensor = torch.from_numpy(
                        np.stack(xs).astype(np.float32)
                    ).to(self.device)

                    # -------------------------------------------------
                    # IMPORTANT:
                    # Anchor must be generated with the historical
                    # State Expert assignment saved for these samples.
                    # -------------------------------------------------
                    if hasattr(
                            self.model,
                            "set_forced_state_ids",
                    ):
                        batch_state_ids = (
                            None
                            if entry.state_ids is None
                            else entry.state_ids[start:end].tolist()
                        )

                        self.model.set_forced_state_ids(
                            batch_state_ids
                        )

                    pred = self.base._model_forward(
                        self.base._transform_x(x_tensor),
                        adjacency,
                    )

                    anchors.append(
                        pred.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

        finally:
            # Clear the historical replay context so it cannot leak
            # into later current-task training/evaluation.
            if hasattr(
                    self.model,
                    "set_forced_state_ids",
            ):
                self.model.set_forced_state_ids(
                    previous_forced_state_ids
                )

            self.model.train(was_training)

            if routing_updates is not None:
                self.model.set_routing_updates(
                    bool(routing_updates)
                )

        anchor_pred = np.concatenate(
            anchors,
            axis=0,
        )

        if len(anchor_pred) != len(entry.indices):
            raise RuntimeError(
                "anchor predictions do not align with replay samples"
            )

        return anchor_pred

    def _refresh_domain_prototypes_from_memory(self) -> None:
        """
        Rebuild existing Domain prototypes in the current backbone
        representation space using task-balanced replay memory.

        Important:
        - Do NOT reroute historical tasks.
        - Do NOT change entry.domain_id.
        - Do NOT change historical state_ids.
        - Do NOT create/delete Domain Experts.
        - Only refresh prototype vectors.
        """
        if self.method != "expert_replay_distill":
            return

        if not self.memory.entries:
            return

        router = getattr(self.model, "regime_router", None)

        if router is None:
            return

        if not hasattr(self.model, "domain_descriptor_samples"):
            return

        if not getattr(router, "domain_prototypes", None):
            return

        was_training = self.model.training

        current_standardizer_dataset = (
            self.active_standardizer_dataset
        )

        routing_updates = getattr(
            self.model,
            "routing_updates_enabled",
            None,
        )

        original_domain_id = getattr(
            self.model,
            "active_domain_id",
            None,
        )

        original_task_id = getattr(
            self.model,
            "active_task_id",
            None,
        )

        descriptor_sums: dict[int, torch.Tensor] = {}
        descriptor_counts: dict[int, int] = {}

        batch_size = int(
            self.config.get("batch_size", 64)
        )

        self.model.eval()

        if hasattr(self.model, "set_routing_updates"):
            self.model.set_routing_updates(False)

        try:
            with torch.no_grad():
                for entry in self.memory.entries:

                    if entry.domain_id is None:
                        continue

                    if len(entry.indices) == 0:
                        continue

                    entry_dataset_name = (
                        entry.dataset_name
                        if entry.dataset_name is not None
                        else self._dataset_name(
                            entry.dataset,
                            task_name=entry.name,
                        )
                    )

                    self._activate_standardizer_by_name(
                        entry_dataset_name
                    )

                    domain_id = int(entry.domain_id)

                    if (
                            domain_id < 0
                            or domain_id >= len(router.domain_prototypes)
                    ):
                        raise IndexError(
                            "replay entry refers to unavailable Domain "
                            f"prototype: task={entry.name} "
                            f"domain={domain_id} "
                            f"num_domains={len(router.domain_prototypes)}"
                        )

                    # Prepare node/model dimensions only.
                    # No new Domain routing is performed here.
                    self.base._prepare_model_for_dataset(
                        entry.dataset,
                        model=self.model,
                    )

                    for start in range(
                            0,
                            len(entry.indices),
                            batch_size,
                    ):
                        end = min(
                            start + batch_size,
                            len(entry.indices),
                        )

                        xs = []

                        for index in entry.indices[start:end]:
                            x, _, _ = unpack_sample(
                                entry.dataset[int(index)]
                            )
                            xs.append(x)

                        if not xs:
                            continue

                        x_tensor = torch.from_numpy(
                            np.stack(xs).astype(np.float32)
                        ).to(self.device)

                        x_tensor = self.base._transform_x(
                            x_tensor
                        )

                        # [B, D]
                        descriptor_samples = (
                            self.model.domain_descriptor_samples(
                                x_tensor
                            )
                        )

                        current_sum = (
                            descriptor_samples
                            .detach()
                            .sum(dim=0)
                            .cpu()
                        )

                        current_count = int(
                            descriptor_samples.shape[0]
                        )

                        if domain_id not in descriptor_sums:
                            descriptor_sums[domain_id] = (
                                current_sum
                            )
                            descriptor_counts[domain_id] = (
                                current_count
                            )
                        else:
                            descriptor_sums[domain_id] += (
                                current_sum
                            )
                            descriptor_counts[domain_id] += (
                                current_count
                            )

            refreshed_domains: list[int] = []

            for domain_id in sorted(descriptor_sums):

                count = descriptor_counts[domain_id]

                if count <= 0:
                    continue

                centroid = (
                        descriptor_sums[domain_id]
                        / float(count)
                )

                router.domain_prototypes[domain_id] = (
                    router._normalize_descriptor(
                        centroid
                    )
                )

                refreshed_domains.append(domain_id)

            refresh_counts = {
                domain_id: descriptor_counts[domain_id]
                for domain_id in refreshed_domains
            }

            print(
                "[Domain Prototype Refresh] "
                f"domains={refreshed_domains} "
                f"counts={refresh_counts}",
                flush=True,
            )

        finally:
            # Refreshing prototypes must not change which Domain
            # remains active for the current task.
            if (
                    original_domain_id is not None
                    and hasattr(
                self.model,
                "activate_recorded_domain",
            )
            ):
                self.model.activate_recorded_domain(
                    int(original_domain_id),
                    task_id=original_task_id,
                )

            if routing_updates is not None:
                self.model.set_routing_updates(
                    bool(routing_updates)
                )

            self.model.train(was_training)

            if current_standardizer_dataset is not None:
                self._activate_standardizer_by_name(
                    current_standardizer_dataset
                )

    def _refresh_state_prototypes_from_memory(self) -> None:
        """
        Rebuild existing State prototypes in the current latent space using
        replay-memory samples and their immutable historical State assignments.

        Important:
        - Do NOT reroute historical replay samples.
        - Do NOT change entry.domain_id.
        - Do NOT change entry.state_ids.
        - Do NOT create/delete State Experts.
        - Do NOT change State Expert counts.
        - Do NOT change state usage counts.
        - Only refresh existing State prototype vectors.

        State descriptor:
            current backbone
            -> recorded Domain Expert
            -> domain_hidden
            -> mean over nodes
            -> State descriptor
        """
        if self.method != "expert_replay_distill":
            return

        if not self.memory.entries:
            return

        # State Expert may intentionally be disabled in an ablation.
        if hasattr(self.model, "enable_state_expert"):
            if not bool(self.model.enable_state_expert):
                return

        router = getattr(
            self.model,
            "regime_router",
            None,
        )

        if router is None:
            return

        if not hasattr(router, "state_prototypes"):
            return

        if not router.state_prototypes:
            return

        if not hasattr(self.model, "encode"):
            return

        if not hasattr(self.model, "domain_experts"):
            return

        if not hasattr(router, "pool_windows"):
            return

        was_training = self.model.training

        current_standardizer_dataset = (
            self.active_standardizer_dataset
        )

        routing_updates = getattr(
            self.model,
            "routing_updates_enabled",
            None,
        )

        original_domain_id = getattr(
            self.model,
            "active_domain_id",
            None,
        )

        original_task_id = getattr(
            self.model,
            "active_task_id",
            None,
        )

        # Key = (domain_id, state_id)
        descriptor_sums: dict[
            tuple[int, int],
            torch.Tensor,
        ] = {}

        descriptor_counts: dict[
            tuple[int, int],
            int,
        ] = {}

        batch_size = int(
            self.config.get(
                "batch_size",
                64,
            )
        )

        self.model.eval()

        if hasattr(
                self.model,
                "set_routing_updates",
        ):
            self.model.set_routing_updates(False)

        try:
            with torch.no_grad():

                for entry in self.memory.entries:

                    if entry.domain_id is None:
                        continue

                    if len(entry.indices) == 0:
                        continue

                    entry_dataset_name = (
                        entry.dataset_name
                        if entry.dataset_name is not None
                        else self._dataset_name(
                            entry.dataset,
                            task_name=entry.name,
                        )
                    )

                    self._activate_standardizer_by_name(
                        entry_dataset_name
                    )

                    if entry.state_ids is None:
                        raise RuntimeError(
                            "State prototype refresh requires "
                            "historical state assignments: "
                            f"task={entry.name}"
                        )

                    domain_id = int(
                        entry.domain_id
                    )

                    state_ids = np.asarray(
                        entry.state_ids,
                        dtype=np.int64,
                    ).reshape(-1)

                    if len(state_ids) != len(
                            entry.indices
                    ):
                        raise RuntimeError(
                            "replay State assignments do not "
                            "align with replay samples during "
                            "State prototype refresh: "
                            f"task={entry.name} "
                            f"indices={len(entry.indices)} "
                            f"state_ids={len(state_ids)}"
                        )

                    if (
                            domain_id < 0
                            or domain_id
                            >= len(
                        router.state_prototypes
                    )
                    ):
                        raise IndexError(
                            "replay entry refers to "
                            "unavailable State prototype bank: "
                            f"task={entry.name} "
                            f"domain={domain_id} "
                            f"num_domains="
                            f"{len(router.state_prototypes)}"
                        )

                    num_states = len(
                        router.state_prototypes[
                            domain_id
                        ]
                    )

                    if num_states == 0:
                        continue

                    state_banks = getattr(
                        self.model,
                        "state_expert_banks",
                        None,
                    )

                    if state_banks is not None:
                        if domain_id >= len(
                                state_banks
                        ):
                            raise RuntimeError(
                                "State Expert bank is missing "
                                "for recorded Domain: "
                                f"domain={domain_id}"
                            )

                        if (
                                len(
                                    state_banks[
                                        domain_id
                                    ]
                                )
                                != num_states
                        ):
                            raise RuntimeError(
                                "State Expert bank and "
                                "State prototype memory "
                                "are out of sync: "
                                f"domain={domain_id} "
                                f"experts="
                                f"{len(state_banks[domain_id])} "
                                f"prototypes={num_states}"
                            )

                    # Only prepare tensor/model dimensions.
                    # Do not perform new Domain routing.
                    self.base._prepare_model_for_dataset(
                        entry.dataset,
                        model=self.model,
                    )

                    # Explicitly restore the immutable
                    # historical Domain context.
                    if hasattr(
                            self.model,
                            "activate_recorded_domain",
                    ):
                        self.model.activate_recorded_domain(
                            domain_id,
                            task_id=entry.name,
                        )

                    for start in range(
                            0,
                            len(entry.indices),
                            batch_size,
                    ):
                        end = min(
                            start + batch_size,
                            len(entry.indices),
                        )

                        xs = []

                        for index in entry.indices[
                                     start:end
                                     ]:
                            x, _, _ = unpack_sample(
                                entry.dataset[
                                    int(index)
                                ]
                            )
                            xs.append(x)

                        if not xs:
                            continue

                        x_tensor = torch.from_numpy(
                            np.stack(xs).astype(
                                np.float32
                            )
                        ).to(self.device)

                        x_tensor = (
                            self.base._transform_x(
                                x_tensor
                            )
                        )

                        # -------------------------------------------------
                        # IMPORTANT:
                        # Compute State descriptors directly.
                        #
                        # Do NOT call normal model forward(), because that
                        # would invoke State routing again.
                        # -------------------------------------------------
                        backbone_hidden = (
                            self.model.encode(
                                x_tensor
                            )
                        )

                        domain_hidden = (
                            self.model.domain_experts(
                                backbone_hidden,
                                domain_id,
                            )
                        )

                        state_descriptors = (
                            router.pool_windows(
                                domain_hidden
                            )
                        )

                        if (
                                state_descriptors.ndim != 2
                        ):
                            raise RuntimeError(
                                "State descriptor refresh "
                                "expected [B,D], got "
                                f"{tuple(state_descriptors.shape)}"
                            )

                        batch_state_ids = (
                            state_ids[start:end]
                        )

                        if (
                                int(
                                    state_descriptors.shape[0]
                                )
                                != len(batch_state_ids)
                        ):
                            raise RuntimeError(
                                "State descriptors do not "
                                "align with historical "
                                "State assignments: "
                                f"task={entry.name} "
                                f"descriptors="
                                f"{state_descriptors.shape[0]} "
                                f"state_ids="
                                f"{len(batch_state_ids)}"
                            )

                        descriptors_cpu = (
                            state_descriptors
                            .detach()
                            .float()
                            .cpu()
                        )

                        for descriptor, raw_state_id in zip(
                                descriptors_cpu,
                                batch_state_ids,
                        ):
                            state_id = int(
                                raw_state_id
                            )

                            if (
                                    state_id < 0
                                    or state_id
                                    >= num_states
                            ):
                                raise IndexError(
                                    "replay entry refers to "
                                    "unavailable State "
                                    "prototype: "
                                    f"task={entry.name} "
                                    f"domain={domain_id} "
                                    f"state={state_id} "
                                    f"num_states={num_states}"
                                )

                            key = (
                                domain_id,
                                state_id,
                            )

                            if (
                                    key
                                    not in descriptor_sums
                            ):
                                descriptor_sums[
                                    key
                                ] = (
                                    descriptor.clone()
                                )

                                descriptor_counts[
                                    key
                                ] = 1

                            else:
                                descriptor_sums[
                                    key
                                ] += descriptor

                                descriptor_counts[
                                    key
                                ] += 1

            refreshed_pairs: list[
                tuple[int, int]
            ] = []

            for key in sorted(
                    descriptor_sums
            ):
                domain_id, state_id = key

                count = descriptor_counts[
                    key
                ]

                if count <= 0:
                    continue

                centroid = (
                        descriptor_sums[key]
                        / float(count)
                )

                router.state_prototypes[
                    domain_id
                ][state_id] = (
                    router._normalize_descriptor(
                        centroid
                    )
                )

                refreshed_pairs.append(
                    key
                )

            # Existing State prototypes for which replay
            # memory contains no sample are intentionally
            # left unchanged.
            existing_pairs = [
                (
                    domain_id,
                    state_id,
                )
                for domain_id, prototypes
                in enumerate(
                    router.state_prototypes
                )
                for state_id
                in range(len(prototypes))
            ]

            missing_pairs = [
                key
                for key in existing_pairs
                if key not in descriptor_counts
            ]

            refresh_counts = {
                f"{domain_id}:{state_id}":
                    descriptor_counts[
                        (
                            domain_id,
                            state_id,
                        )
                    ]
                for domain_id, state_id
                in refreshed_pairs
            }

            print(
                "[State Prototype Refresh] "
                f"pairs={refreshed_pairs} "
                f"counts={refresh_counts} "
                f"missing={missing_pairs}",
                flush=True,
            )

        finally:
            # Restore the Domain that was active before
            # prototype refresh.
            if (
                    original_domain_id is not None
                    and hasattr(
                self.model,
                "activate_recorded_domain",
            )
            ):
                self.model.activate_recorded_domain(
                    int(original_domain_id),
                    task_id=original_task_id,
                )

            if routing_updates is not None:
                self.model.set_routing_updates(
                    bool(routing_updates)
                )

            self.model.train(
                was_training
            )

            if current_standardizer_dataset is not None:
                self._activate_standardizer_by_name(
                    current_standardizer_dataset
                )

    def _configure_model(self, model: nn.Module, dataset: Dataset) -> None:
        self.base._prepare_model_for_dataset(dataset, model=model)
        self.base.activate_routing_for_dataset(dataset, model=model)

    @staticmethod
    def _forward_model(
        model: nn.Module,
        x: torch.Tensor,
        adjacency: torch.Tensor | None,
    ) -> torch.Tensor:
        try:
            return model(x, adj=adjacency)
        except TypeError:
            return model(x)


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return (prediction[mask] - target[mask]).square().mean()


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return (prediction[mask] - target[mask]).abs().mean()


def normalize_method(method: str) -> str:
    normalized = str(method).lower().replace("-", "_")
    aliases = {
        "sequential_finetuning": "sequential",
        "fine_tuning": "sequential",
        "der++": "derpp",
        "tracl": "expert_replay_distill",
        "trafficl": "expert_replay_distill",
        "ours": "expert_replay_distill",
    }
    return aliases.get(normalized, normalized)
