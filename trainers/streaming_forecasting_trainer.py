from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from utils.checkpointing import atomic_torch_save, try_load_torch
from utils.progress import LogProgressBar


@dataclass
class StreamingStandardizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    # First-TRAIN statistics use raw targets and the same validity mask as loss.
    # Defaults preserve loading/constructing standardizers from older runs.
    target_zero_fraction: float | None = None
    target_valid_count: int = 0

    @classmethod
    def fit(
        cls,
        dataset: Dataset,
        *,
        target_mode: str,
        flow_feature_index: int,
        max_windows: int = 2048,
        eps: float = 1e-6,
    ) -> "StreamingStandardizer":
        return cls.fit_many(
            [dataset],
            target_mode=target_mode,
            flow_feature_index=flow_feature_index,
            max_windows_per_dataset=max_windows,
            eps=eps,
        )

    @classmethod
    def fit_many(
        cls,
        datasets: list[Dataset],
        *,
        target_mode: str,
        flow_feature_index: int,
        max_windows_per_dataset: int = 2048,
        eps: float = 1e-6,
    ) -> "StreamingStandardizer":
        x_sum = None
        x_sumsq = None
        x_count = 0
        y_sum = 0.0
        y_sumsq = 0.0
        y_count = 0
        raw_target_zero_count = 0
        total_datasets = len(datasets)
        for dataset_index, dataset in enumerate(datasets, start=1):
            count = min(len(dataset), max_windows_per_dataset)
            indices = evenly_spaced_indices(len(dataset), count)
            task_name = getattr(getattr(dataset, "period", None), "name", str(dataset_index))
            print(
                f"standardizer task={dataset_index}/{total_datasets} "
                f"name={task_name} sampled_windows={count}",
                flush=True,
            )
            for index in indices:
                x, y, y_mask = unpack_sample(dataset[index])
                target = target_array(
                    x[None, ...],
                    y[None, ...],
                    target_mode=target_mode,
                    flow_feature_index=flow_feature_index,
                )[0]
                valid_target = target[np.asarray(y_mask, dtype=bool)]
                valid_raw_target = np.asarray(y)[np.asarray(y_mask, dtype=bool)]
                raw_target_zero_count += int(np.count_nonzero(valid_raw_target == 0.0))
                x_flat = x.reshape(-1, x.shape[-1])
                if x_sum is None:
                    x_sum = x_flat.sum(axis=0, dtype=np.float64)
                    x_sumsq = np.square(x_flat.astype(np.float64)).sum(axis=0)
                else:
                    x_sum += x_flat.sum(axis=0, dtype=np.float64)
                    x_sumsq += np.square(x_flat.astype(np.float64)).sum(axis=0)
                x_count += len(x_flat)

                y_sum += float(valid_target.sum(dtype=np.float64))
                y_sumsq += float(np.square(valid_target.astype(np.float64)).sum())
                y_count += valid_target.size

        assert x_sum is not None and x_sumsq is not None
        if y_count == 0:
            raise ValueError("standardizer received no valid forecasting targets")
        x_mean = x_sum / max(x_count, 1)
        x_var = x_sumsq / max(x_count, 1) - x_mean**2
        x_std = np.sqrt(np.maximum(x_var, eps)).reshape(1, 1, 1, -1).astype(np.float32)
        x_mean = x_mean.reshape(1, 1, 1, -1).astype(np.float32)

        y_mean = y_sum / max(y_count, 1)
        y_var = y_sumsq / max(y_count, 1) - y_mean**2
        y_std = float(np.sqrt(max(y_var, eps)))
        return cls(x_mean=x_mean, x_std=x_std, y_mean=float(y_mean), y_std=y_std,
                   target_zero_fraction=raw_target_zero_count / y_count,
                   target_valid_count=y_count)


def evenly_spaced_indices(length: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count >= length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=count, dtype=np.int64)


class StreamingForecastingTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
        *,
        output_dir: str | Path,
        checkpoint_dir: str | Path | None = None,
    ):
        self.model = model
        self.config = dict(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else self.output_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(str(self.config.get("device", "auto")))
        self.model.to(self.device)
        self.loss_fn = self._build_loss()
        self.active_loss_name = str(self.config.get("loss", "huber")).lower()
        self.lambda_harm = float(self.config.get("lambda_harm", 0.1))
        self.lambda_safe = float(self.config.get("lambda_safe", 0.2))
        self.lambda_gate = float(self.config.get("lambda_gate", 0.0))
        self.lambda_gate_sup = float(self.config.get("lambda_gate_sup", 0.1))
        self.gate_supervision_margin = float(self.config.get("gate_supervision_margin", 0.01))
        self.lambda_mae = float(self.config.get("lambda_mae", 0.5))
        self.lambda_bias = float(self.config.get("lambda_bias", 0.0))
        self.target_mode = str(self.config.get("target_mode", "residual"))
        self.flow_feature_index = int(self.config.get("flow_feature_index", 0))
        self.standardizer: StreamingStandardizer | None = None
        self.residual_mean: float = 0.0
        self.residual_std: float = 1.0
        self._routing_domains: dict[tuple[int, int], int] = {}

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        *,
        refit_standardizer: bool = True,
        checkpoint_name: str = "best_model_streaming.pt",
    ) -> dict[str, Any]:
        if self.standardizer is None or refit_standardizer:
            self.standardizer = StreamingStandardizer.fit(
                train_dataset,
                target_mode=self.target_mode,
                flow_feature_index=self.flow_feature_index,
                max_windows=int(self.config.get("standardizer_max_windows", 2048)),
            )
        self._activate_standardizer_loss()
        if self.target_mode == "residual" and bool(self.config.get("residual_normalization", True)):
            self.residual_mean, self.residual_std = self._fit_residual_stats(train_dataset)
        self._prepare_model_for_dataset(train_dataset)
        self._configure_model_residual_normalization()
        train_loader = self._loader(train_dataset, shuffle=True)
        val_loader = self._loader(val_dataset, shuffle=False)
        train_adj = self._adjacency_tensor(train_dataset)
        val_adj = self._adjacency_tensor(val_dataset)

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
        if last_state is not None:
            self.model.load_state_dict(last_state["model_state"])
            self.standardizer = last_state["standardizer"]
            self._activate_standardizer_loss()
            self.residual_mean = float(last_state.get("residual_mean", self.residual_mean))
            self.residual_std = float(last_state.get("residual_std", self.residual_std))
            history = list(last_state["history"])
            best_val = float(last_state["best_val_loss"])
            epochs_without_improvement = int(last_state["epochs_without_improvement"])
            start_epoch = int(last_state["next_epoch"])
            print(f"Resume independent task from epoch={start_epoch}/{epochs}", flush=True)

        self._prepare_model_for_dataset(train_dataset)
        task_id = self._dataset_task_id(train_dataset)
        if last_state is not None and getattr(self.model, "active_domain_id", None) is not None:
            self._remember_routing_domain(
                self.model,
                (train_dataset, val_dataset),
                int(self.model.active_domain_id),
                task_id=task_id,
            )
        else:
            decision = self.prepare_routing_for_dataset(
                train_dataset,
                task_id=task_id,
                allow_create=True,
                update_prototype=True,
                record_assignment=True,
            )
            if decision is not None:
                self._remember_routing_domain(
                    self.model,
                    (val_dataset,),
                    int(decision.selected_id),
                    task_id=task_id,
                )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.get("lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        if last_state is not None:
            if last_state.get("optimizer_state_by_name") is not None:
                self._load_optimizer_state_by_name(
                    optimizer,
                    self.model,
                    last_state["optimizer_state_by_name"],
                )
            else:
                optimizer.load_state_dict(last_state["optimizer_state"])

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self._train_epoch(
                train_loader,
                optimizer,
                train_adj,
                progress_label=f"TRAIN epoch={epoch:03d}/{epochs:03d}",
            )
            val_loss = self._loss_epoch(
                val_loader,
                val_adj,
                progress_label=f"VAL epoch={epoch:03d}/{epochs:03d}",
            )
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val - min_delta:
                best_val = val_loss
                epochs_without_improvement = 0
                atomic_torch_save(
                    {
                        "model_state": self.model.state_dict(),
                        "standardizer": self.standardizer,
                        "residual_mean": self.residual_mean,
                        "residual_std": self.residual_std,
                        "config": self.config,
                        "history": history,
                        "task_id": task_id,
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            atomic_torch_save(
                {
                    "next_epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "optimizer_state_by_name": self._optimizer_state_by_name(
                        optimizer, self.model
                    ),
                    "standardizer": self.standardizer,
                    "residual_mean": self.residual_mean,
                    "residual_std": self.residual_std,
                    "history": history,
                    "best_val_loss": best_val,
                    "epochs_without_improvement": epochs_without_improvement,
                    "task_id": task_id,
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
        if getattr(self.model, "active_domain_id", None) is not None:
            self._remember_routing_domain(
                self.model,
                (train_dataset, val_dataset),
                int(self.model.active_domain_id),
                task_id=task_id,
            )
        return {
            "history": history,
            "best_val_loss": best_val,
            "checkpoint": str(best_path),
            "routing": self._routing_summary(self.model),
        }

    def evaluate(self, dataset: Dataset) -> dict[str, float]:
        self._prepare_model_for_dataset(dataset)
        self._configure_model_residual_normalization()

        task_id = self._dataset_task_id(dataset)

        recorded_domain_id = None

        if hasattr(
                self.model,
                "recorded_domain_for_task",
        ):
            recorded_domain_id = (
                self.model.recorded_domain_for_task(
                    task_id
                )
            )

        if recorded_domain_id is not None:
            # Historical task:
            # restore the Domain Expert originally assigned
            # when this task was first learned.
            self._remember_routing_domain(
                self.model,
                (dataset,),
                int(recorded_domain_id),
                task_id=task_id,
            )

            print(
                "[Historical Domain Restore] "
                f"task_id={task_id} "
                f"domain_id={recorded_domain_id}",
                flush=True,
            )

        else:
            # Unseen task:
            # inference-time routing is still allowed,
            # but no new Domain is created and no prototype is updated.
            self.prepare_routing_for_dataset(
                dataset,
                task_id=task_id,
                allow_create=False,
                update_prototype=False,
                record_assignment=False,
            )
        loader = self._loader(dataset, shuffle=False)
        adj = self._adjacency_tensor(dataset)
        total_abs = 0.0
        total_sq = 0.0
        total_count = 0
        total_mape = 0.0
        mape_count = 0
        total_target_abs = 0.0
        total_target = 0.0
        total_target_sq = 0.0
        total_persistence_abs = 0.0
        total_backbone_abs = 0.0
        total_backbone_count = 0
        total_prediction_bias = 0.0
        total_prediction_sq = 0.0
        gate_snapshots: list[dict[str, Any]] = []

        self.model.eval()
        with torch.no_grad():
            for batch in self._limited_batches(loader, "max_eval_batches"):
                x, y, y_mask = unpack_batch(batch)
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                y_mask = y_mask.to(self.device).bool()
                pred = self._predict_raw(x, adj)
                components_raw = self._component_predictions_raw(x, y, y_mask, adj)
                error = pred - y
                valid_y = y[y_mask]
                valid_error = error[y_mask]
                persistence = last_value_forecast(x, self.flow_feature_index, y.shape[1])
                persistence_error = persistence - y
                total_abs += float(torch.abs(valid_error).sum().item())
                total_sq += float(torch.square(valid_error).sum().item())
                total_target_abs += float(torch.abs(valid_y).sum().item())
                total_target += float(valid_y.sum().item())
                total_target_sq += float(torch.square(valid_y).sum().item())
                total_persistence_abs += float(torch.abs(persistence_error[y_mask]).sum().item())
                total_prediction_bias += float(valid_error.sum().item())
                total_prediction_sq += float(torch.square(pred[y_mask]).sum().item())
                if components_raw is not None:
                    backbone_error = components_raw["backbone_forecast"] - y
                    total_backbone_abs += float(torch.abs(backbone_error[y_mask]).sum().item())
                    total_backbone_count += int(valid_y.numel())
                    gate_snapshots.append(components_raw.get("gates", {}))
                total_count += int(valid_y.numel())
                mask = y_mask & (torch.abs(y) > 1.0)
                if bool(mask.any()):
                    total_mape += float((torch.abs(error[mask] / y[mask])).sum().item())
                    mape_count += int(mask.sum().item())

        denominator = max(total_count, 1)
        mae_value = total_abs / denominator
        mse_value = total_sq / denominator
        rmse_value = float(np.sqrt(mse_value))
        target_mean = total_target / denominator
        target_variance = max(total_target_sq / denominator - target_mean**2, 0.0)
        target_std = float(np.sqrt(target_variance))
        persistence_mae = total_persistence_abs / denominator
        relative_error = (
            mae_value / persistence_mae if persistence_mae > 1e-8 else float("nan")
        )
        backbone_mae = total_backbone_abs / max(total_backbone_count, 1) if total_backbone_count else float("nan")
        backbone_relative = backbone_mae / persistence_mae if persistence_mae > 1e-8 else float("nan")
        prediction_bias = total_prediction_bias / denominator
        prediction_std = float(np.sqrt(max(total_prediction_sq / denominator - (total_target / denominator + prediction_bias) ** 2, 0.0)))
        gate_summary = self._summarize_gate_snapshots(gate_snapshots)
        result = {
            "mae": mae_value,
            "mse": mse_value,
            "rmse": rmse_value,
            "mape": 100.0 * total_mape / max(mape_count, 1),
            "wape": 100.0 * total_abs / total_target_abs if total_target_abs > 1e-8 else float("nan"),
            "nrmse": rmse_value / target_std if target_std > 1e-8 else float("nan"),
            "target_std": target_std,
            "valid_target_count": total_count,
            "persistence_mae": persistence_mae,
            "relative_mae": relative_error,
            "skill": 1.0 - relative_error if np.isfinite(relative_error) else float("nan"),
            "backbone_mae": backbone_mae,
            "expert_mae": mae_value,
            "backbone_skill": 1.0 - backbone_relative if np.isfinite(backbone_relative) else float("nan"),
            "expert_skill": 1.0 - relative_error if np.isfinite(relative_error) else float("nan"),
            "expert_delta_mae": mae_value - backbone_mae if np.isfinite(backbone_mae) else float("nan"),
            "prediction_bias": prediction_bias,
            "prediction_std_ratio": prediction_std / target_std if target_std > 1e-8 else float("nan"),
        }
        result.update(gate_summary)
        return result

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        adj: torch.Tensor | None,
        *,
        progress_label: str | None = None,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        total_items = 0
        total_batches = self._batch_count(loader, "max_train_batches")
        progress = LogProgressBar(progress_label, total_batches) if progress_label else None
        for batch_index, batch in enumerate(
            self._limited_batches(loader, "max_train_batches"),
            start=1,
        ):
            x, y, y_mask = unpack_batch(batch)
            x = x.to(self.device).float()
            y = y.to(self.device).float()
            y_mask = y_mask.to(self.device).bool()
            valid_count = int(y_mask.sum().item())
            if valid_count == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            x_scaled = self._transform_x(x)
            pred, components = self._model_forward_with_components(x_scaled, adj)
            self._sync_optimizer_parameters(optimizer, self.model)
            target = self._transform_y(self._target_array(x, y))
            forecast_loss = self._masked_loss(pred, target, y_mask)
            harm_loss = self._do_no_harm_loss(components, target, y_mask)
            safe_loss = self._persistence_safe_loss(pred, target, y_mask)
            gate_loss = self._gate_regularization(components)
            gate_sup_loss = self._gate_supervision_loss(components, target, y_mask)
            bias_loss = self._bias_loss(pred, target, y_mask)
            loss = (
                forecast_loss
                + self.lambda_harm * harm_loss
                + self.lambda_safe * safe_loss
                + self.lambda_gate * gate_loss
                + self.lambda_gate_sup * gate_sup_loss
                + self.lambda_bias * bias_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("grad_clip", 1.0)))
            optimizer.step()
            total_loss += float(loss.item()) * valid_count
            total_items += valid_count
            if progress is not None:
                progress.update(batch_index, loss=total_loss / max(total_items, 1))
        return total_loss / max(total_items, 1)

    def _loss_epoch(
        self,
        loader: DataLoader,
        adj: torch.Tensor | None,
        *,
        progress_label: str | None = None,
    ) -> float:
        self.model.eval()
        validation_loss = self.config.get("validation_loss")
        if validation_loss not in (None, "mae"):
            raise ValueError("validation_loss must be unset or 'mae'")
        total_loss = 0.0
        total_items = 0
        total_batches = self._batch_count(loader, "max_val_batches")
        progress = LogProgressBar(progress_label, total_batches) if progress_label else None
        with torch.no_grad():
            for batch_index, batch in enumerate(
                self._limited_batches(loader, "max_val_batches"),
                start=1,
            ):
                x, y, y_mask = unpack_batch(batch)
                x = x.to(self.device).float()
                y = y.to(self.device).float()
                y_mask = y_mask.to(self.device).bool()
                valid_count = int(y_mask.sum().item())
                if valid_count == 0:
                    continue
                pred = self._model_forward(self._transform_x(x), adj)
                target = self._transform_y(self._target_array(x, y))
                # Select checkpoints using the same support projection deployed
                # at inference. Training retains its unconstrained surrogate so
                # negative outputs still receive a corrective gradient.
                if self.target_mode == "direct" and self.config.get("nonnegative_predictions", False):
                    pred = self._transform_y(self._inverse_y(pred))
                loss = (torch.abs(pred - target)[y_mask].mean()
                        if validation_loss == "mae"
                        else self._masked_loss(pred, target, y_mask))
                total_loss += float(loss.item()) * valid_count
                total_items += valid_count
                if progress is not None:
                    progress.update(batch_index, loss=total_loss / max(total_items, 1))
        return total_loss / max(total_items, 1)

    def _predict_raw(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        pred_scaled = self._model_forward(self._transform_x(x), adj)
        pred_target = self._inverse_y(pred_scaled)
        if self.target_mode == "residual":
            return last_value_forecast(x, self.flow_feature_index, pred_target.shape[1]) + pred_target
        if self.target_mode == "direct":
            return pred_target
        raise ValueError(f"unsupported target mode: {self.target_mode}")

    def _target_array(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "residual":
            return y - last_value_forecast(x, self.flow_feature_index, y.shape[1])
        if self.target_mode == "direct":
            return y
        raise ValueError(f"unsupported target mode: {self.target_mode}")

    def _transform_x(self, x: torch.Tensor) -> torch.Tensor:
        assert self.standardizer is not None
        mean = torch.as_tensor(self.standardizer.x_mean, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.standardizer.x_std, device=x.device, dtype=x.dtype)
        # The canonical model reverses its window normalization at the output.
        # Thus input flow and supervised output MUST use the same affine scale.
        # Otherwise a constant window is decoded as y_std*(x-x_mean)/x_std+y_mean.
        if self.target_mode == "direct" and hasattr(self.model, "canonical_head"):
            mean = mean.clone()
            std = std.clone()
            mean[..., self.flow_feature_index] = self.standardizer.y_mean
            std[..., self.flow_feature_index] = self.standardizer.y_std
        return (x - mean) / std

    def _transform_y(self, y: torch.Tensor) -> torch.Tensor:
        assert self.standardizer is not None
        if self.target_mode == "residual" and bool(self.config.get("residual_normalization", True)):
            return (y - self.residual_mean) / max(self.residual_std, 1e-6)
        return (y - self.standardizer.y_mean) / self.standardizer.y_std

    def _inverse_y(self, y: torch.Tensor) -> torch.Tensor:
        assert self.standardizer is not None
        if self.target_mode == "residual" and bool(self.config.get("residual_normalization", True)):
            return y * max(self.residual_std, 1e-6) + self.residual_mean
        raw = y * self.standardizer.y_std + self.standardizer.y_mean
        if self.target_mode == "direct" and bool(self.config.get("nonnegative_predictions", False)):
            # Flow/count forecasts have nonnegative support. This deterministic
            # projection is independent of targets and of the persistence baseline.
            raw = raw.clamp_min(0.0)
        return raw

    def _fit_residual_stats(self, dataset: Dataset) -> tuple[float, float]:
        values = []
        max_windows = int(self.config.get("standardizer_max_windows", 2048))
        count = min(len(dataset), max_windows)
        for index in evenly_spaced_indices(len(dataset), count):
            x, y, y_mask = unpack_sample(dataset[int(index)])
            residual = target_array(
                x[None, ...],
                y[None, ...],
                target_mode="residual",
                flow_feature_index=self.flow_feature_index,
            )[0]
            valid = np.asarray(y_mask, dtype=bool)
            values.append(residual[valid].astype(np.float64))
        if not values:
            return 0.0, 1.0
        merged = np.concatenate(values, axis=0)
        if merged.size == 0:
            return 0.0, 1.0
        std = float(np.std(merged))
        return float(np.mean(merged)), max(std, 1e-6)

    def _configure_model_residual_normalization(self) -> None:
        if hasattr(self.model, "set_residual_normalization"):
            self.model.set_residual_normalization(self.residual_mean, self.residual_std)

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        actual_shuffle = shuffle and bool(self.config.get("shuffle_train", True))
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", 64)),
            shuffle=actual_shuffle,
            num_workers=int(self.config.get("num_workers", 0)),
            drop_last=False,
        )

    def _limited_batches(self, loader: DataLoader, key: str):
        limit = self.config.get(key)
        for index, batch in enumerate(loader):
            if limit not in (None, "") and index >= int(limit):
                break
            yield batch

    def _batch_count(self, loader: DataLoader, key: str) -> int:
        limit = self.config.get(key)
        if limit in (None, ""):
            return len(loader)
        return min(len(loader), int(limit))

    def _activate_standardizer_loss(self) -> dict[str, Any]:
        """Restore the objective belonging to this physical dataset's TRAIN data."""
        loss_name = str(self.config.get("loss", "huber")).lower()
        fraction = getattr(self.standardizer, "target_zero_fraction", None)
        count = int(getattr(self.standardizer, "target_valid_count", 0))
        adaptive = bool(self.config.get("adaptive_loss", False))
        threshold = float(self.config.get("sparse_zero_fraction", 0.5))
        if adaptive:
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("sparse_zero_fraction must be finite and between 0 and 1")
            if (fraction is None or not np.isfinite(fraction)
                    or not 0.0 <= fraction <= 1.0 or count <= 0):
                raise ValueError("adaptive_loss requires saved first-TRAIN target_zero_fraction; "
                                 "start a new run to fit the training standardizer")
            loss_name = "mae" if fraction >= threshold else "huber"
        if self.active_loss_name != loss_name:
            self.loss_fn = self._build_loss(loss_name)
            self.active_loss_name = loss_name
        return {"adaptive_loss": adaptive, "loss": loss_name,
                "target_zero_fraction": fraction, "target_valid_count": count,
                "sparse_zero_fraction": threshold,
                "statistics_source": "first_task_train_standardizer_samples"}

    def _build_loss(self, loss_name: str | None = None) -> nn.Module:
        loss_name = str(self.config.get("loss", "huber") if loss_name is None else loss_name).lower()
        if loss_name == "mse":
            return nn.MSELoss(reduction="none")
        if loss_name in {"mae", "l1"}:
            return nn.L1Loss(reduction="none")
        if loss_name in {"huber", "smooth_l1"}:
            return nn.SmoothL1Loss(
                beta=float(self.config.get("huber_beta", 1.0)),
                reduction="none",
            )
        raise ValueError(f"unsupported loss: {loss_name}")

    def _masked_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        element_loss = self.loss_fn(prediction, target)
        loss = element_loss[mask].mean()
        if self.lambda_mae > 0.0 and not isinstance(self.loss_fn, nn.L1Loss):
            loss = loss + self.lambda_mae * torch.abs(prediction - target)[mask].mean()
        return loss

    def _do_no_harm_loss(
        self,
        components: Mapping[str, Any] | None,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not components or "backbone_forecast" not in components:
            return target.new_tensor(0.0)
        backbone_prediction = components["backbone_forecast"]
        if not isinstance(backbone_prediction, torch.Tensor):
            return target.new_tensor(0.0)
        expert_prediction = components.get("final_forecast")
        if not isinstance(expert_prediction, torch.Tensor):
            return target.new_tensor(0.0)
        expert_loss = self.loss_fn(expert_prediction, target)[mask].mean()
        backbone_loss = self.loss_fn(backbone_prediction, target)[mask].mean().detach()
        return torch.relu(expert_loss - backbone_loss)

    def _model_forward_with_components(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None,
    ) -> tuple[torch.Tensor, Mapping[str, Any] | None]:
        if hasattr(self.model, "forward_with_components"):
            pred, components = self.model.forward_with_components(x)
            components = dict(components)
            components["final_forecast"] = pred
            return pred, components
        pred = self._model_forward(x, adj)
        return pred, None

    def _component_predictions_raw(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        adj: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        if not hasattr(self.model, "forward_with_components"):
            return None
        x_scaled = self._transform_x(x)
        _, components = self.model.forward_with_components(x_scaled)
        if "backbone_forecast" not in components:
            return None
        backbone_target = self._inverse_y(components["backbone_forecast"])
        expert_target = self._inverse_y(components.get("final_forecast", self.model(x_scaled)))
        candidate_target = None
        if "adapted_residual_raw" in components:
            candidate_target = self._inverse_y(
                self._persistence_normalized_prediction(components["adapted_residual_raw"])
                + components["adapted_residual_raw"]
            )
        if self.target_mode == "residual":
            persistence = last_value_forecast(x, self.flow_feature_index, backbone_target.shape[1])
            backbone_target = persistence + backbone_target
            expert_target = persistence + expert_target
            if candidate_target is not None:
                candidate_target = persistence + candidate_target
        gates = {
            "mean_safety_gate": float(components["safety_gate"].detach().mean().cpu()) if "safety_gate" in components else float("nan"),
            "mean_sample_gate": float(components["sample_gate"].detach().mean().cpu()) if "sample_gate" in components else float("nan"),
            "domain_gate": float(components["domain_gate"].detach().cpu()) if "domain_gate" in components else float("nan"),
            "state_gate": float(components["state_gate"].detach().cpu()) if "state_gate" in components else float("nan"),
            "residual_mean": float(self.residual_mean),
            "residual_std": float(self.residual_std),
        }
        if candidate_target is not None:
            persistence = last_value_forecast(x, self.flow_feature_index, expert_target.shape[1])
            candidate_mae_b = self._sample_mae_raw(candidate_target, y, y_mask)
            persistence_mae_b = self._sample_mae_raw(persistence, y, y_mask)
            gated_mae_b = self._sample_mae_raw(expert_target, y, y_mask)
            target_1 = candidate_mae_b < persistence_mae_b * (1.0 - self.gate_supervision_margin)
            gates.update(
                {
                    "ungated_candidate_mae": float(candidate_mae_b.mean().cpu()),
                    "gated_model_mae": float(gated_mae_b.mean().cpu()),
                    "fraction_candidate_beats_persistence": float((candidate_mae_b < persistence_mae_b).float().mean().cpu()),
                    "fraction_gated_model_beats_persistence": float((gated_mae_b < persistence_mae_b).float().mean().cpu()),
                    "fraction_gate_target_1": float(target_1.float().mean().cpu()),
                }
            )
            if "sample_gate" in components:
                sample_prob_b = components["sample_gate"].detach().mean(dim=1).reshape(-1)
                pred_1 = sample_prob_b >= 0.5
                gates["gate_supervision_accuracy"] = float((pred_1 == target_1).float().mean().cpu())
                gates["gate_supervision_bce"] = float(
                    F.binary_cross_entropy(
                        sample_prob_b.clamp(1e-6, 1.0 - 1e-6),
                        target_1.float(),
                    ).cpu()
                )
        if "horizon_gate" in components:
            values = components["horizon_gate"].detach().reshape(-1).cpu().tolist()
            for index, value in enumerate(values, start=1):
                gates[f"horizon_gate_{index}"] = float(value)
        return {"backbone_forecast": backbone_target, "expert_forecast": expert_target, "gates": gates}

    @staticmethod
    def _summarize_gate_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, float]:
        if not snapshots:
            return {}
        keys = sorted({key for snapshot in snapshots for key in snapshot})
        summary = {}
        for key in keys:
            values = [float(snapshot[key]) for snapshot in snapshots if key in snapshot and np.isfinite(float(snapshot[key]))]
            if values:
                summary[key] = float(np.mean(values))
        return summary

    def _persistence_normalized_prediction(self, target: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "residual" and bool(self.config.get("residual_normalization", True)):
            value = -float(self.residual_mean) / max(float(self.residual_std), 1e-6)
            return torch.full_like(target, value)
        return torch.zeros_like(target)

    def _sample_mae_normalized(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        error = torch.abs(prediction - target)
        sample_dims = tuple(range(1, error.ndim))
        weights = mask.to(error.dtype)
        numerator = (error * weights).sum(dim=sample_dims)
        denominator = weights.sum(dim=sample_dims).clamp_min(1.0)
        return numerator / denominator

    def _sample_mae_raw(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        error = torch.abs(prediction - target)
        sample_dims = tuple(range(1, error.ndim))
        weights = mask.to(error.dtype)
        numerator = (error * weights).sum(dim=sample_dims)
        denominator = weights.sum(dim=sample_dims).clamp_min(1.0)
        return numerator / denominator

    def _persistence_safe_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        persistence_prediction = self._persistence_normalized_prediction(target)
        model_loss_b = self._sample_mae_normalized(prediction, target, mask)
        persistence_loss_b = self._sample_mae_normalized(persistence_prediction, target, mask).detach()
        return torch.relu(model_loss_b - persistence_loss_b).mean()

    def _gate_supervision_loss(
        self,
        components: Mapping[str, Any] | None,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not components or "sample_gate" not in components or "adapted_residual_raw" not in components:
            return target.new_tensor(0.0)
        persistence_prediction = self._persistence_normalized_prediction(target)
        candidate_prediction = persistence_prediction + components["adapted_residual_raw"]
        candidate_loss_b = self._sample_mae_normalized(candidate_prediction, target, mask).detach()
        persistence_loss_b = self._sample_mae_normalized(persistence_prediction, target, mask).detach()
        gate_target_b = (candidate_loss_b < persistence_loss_b * (1.0 - self.gate_supervision_margin)).to(target.dtype)
        sample_gate = components["sample_gate"]
        if sample_gate.ndim == 3:
            sample_gate = sample_gate.squeeze(-1)
        gate_target = gate_target_b[:, None].expand_as(sample_gate).detach()
        return F.binary_cross_entropy(sample_gate.clamp(1e-6, 1.0 - 1e-6), gate_target)

    def _gate_regularization(self, components: Mapping[str, Any] | None) -> torch.Tensor:
        if not components or "safety_gate" not in components:
            parameter = next(self.model.parameters())
            return parameter.new_tensor(0.0)
        gate = components["safety_gate"]
        return gate.mean() if isinstance(gate, torch.Tensor) else next(self.model.parameters()).new_tensor(0.0)

    def _bias_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if not bool(mask.any()):
            return prediction.new_tensor(0.0)
        pred_raw = self._inverse_y(prediction)
        target_raw = self._inverse_y(target)
        return torch.abs((pred_raw - target_raw)[mask].mean())

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def prepare_routing_for_dataset(
        self,
        dataset: Dataset,
        *,
        task_id: str,
        allow_create: bool,
        update_prototype: bool,
        record_assignment: bool,
        model: nn.Module | None = None,
    ) -> Any | None:
        target_model = self.model if model is None else model
        if not (
            hasattr(target_model, "domain_descriptor_samples")
            and hasattr(target_model, "route_task_descriptor")
        ):
            return None
        if self.standardizer is None:
            raise RuntimeError("a fitted standardizer is required before domain routing")

        self._prepare_model_for_dataset(dataset, model=target_model)
        max_windows = int(getattr(target_model, "domain_descriptor_max_windows", 2048))
        descriptor_count_target = min(len(dataset), max_windows)
        descriptor_indices = evenly_spaced_indices(len(dataset), descriptor_count_target)
        descriptor_dataset: Dataset = Subset(dataset, descriptor_indices.tolist())
        loader = self._loader(descriptor_dataset, shuffle=False)
        descriptor_sum: torch.Tensor | None = None
        descriptor_count = 0
        was_training = target_model.training
        target_model.eval()
        with torch.no_grad():
            for batch in loader:
                x, _, _ = unpack_batch(batch)
                x = self._transform_x(x.to(self.device).float())
                samples = target_model.domain_descriptor_samples(x)
                remaining = max_windows - descriptor_count
                if remaining <= 0:
                    break
                samples = samples[:remaining]
                current = samples.detach().sum(dim=0)
                descriptor_sum = current if descriptor_sum is None else descriptor_sum + current
                descriptor_count += int(samples.shape[0])
                if descriptor_count >= max_windows:
                    break
        target_model.train(was_training)
        if descriptor_sum is None or descriptor_count == 0:
            raise ValueError(f"cannot compute a domain descriptor for empty task {task_id!r}")

        descriptor = descriptor_sum / float(descriptor_count)
        decision = target_model.route_task_descriptor(
            descriptor,
            task_id=task_id,
            allow_create=allow_create,
            update_prototype=update_prototype,
            record_assignment=record_assignment,
        )
        self._remember_routing_domain(
            target_model,
            (dataset,),
            int(decision.selected_id),
            task_id=task_id,
        )
        return decision

    def activate_routing_for_dataset(
        self,
        dataset: Dataset,
        *,
        model: nn.Module | None = None,
        task_id: str | None = None,
    ) -> int | None:
        target_model = self.model if model is None else model
        key = (id(target_model), id(dataset))
        domain_id = self._routing_domains.get(key)
        if domain_id is None or not hasattr(target_model, "activate_recorded_domain"):
            return None
        target_model.activate_recorded_domain(domain_id, task_id=task_id)
        return domain_id

    def _remember_routing_domain(
        self,
        model: nn.Module,
        datasets: tuple[Dataset, ...],
        domain_id: int,
        *,
        task_id: str | None,
    ) -> None:
        for dataset in datasets:
            self._routing_domains[(id(model), id(dataset))] = int(domain_id)
        if hasattr(model, "activate_recorded_domain"):
            model.activate_recorded_domain(int(domain_id), task_id=task_id)

    @staticmethod
    def _dataset_task_id(dataset: Dataset) -> str:
        period = getattr(dataset, "period", None)
        if period is not None and getattr(period, "name", None) is not None:
            return str(period.name)
        for attribute in ("task_name", "name", "dataset_name"):
            value = getattr(dataset, attribute, None)
            if value is not None:
                return str(value)
        return type(dataset).__name__

    @staticmethod
    def _routing_summary(model: nn.Module) -> dict[str, Any] | None:
        if not hasattr(model, "routing_summary"):
            return None
        return dict(model.routing_summary())

    @staticmethod
    def _sync_optimizer_parameters(
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
    ) -> int:
        known = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        new_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in known
        ]
        if new_parameters:
            optimizer.param_groups[0]["params"].extend(new_parameters)
        return len(new_parameters)

    @staticmethod
    def _optimizer_state_by_name(
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        extra_parameters: Mapping[str, nn.Parameter] | None = None,
    ) -> dict[str, Any]:
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        names.update({id(parameter): name for name, parameter in (extra_parameters or {}).items()})
        state = {
            names[id(parameter)]: StreamingForecastingTrainer._copy_optimizer_value(value)
            for parameter, value in optimizer.state.items()
            if id(parameter) in names
        }
        group_options = [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ]
        return {"state": state, "param_group_options": group_options}

    @staticmethod
    def _load_optimizer_state_by_name(
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        saved: Mapping[str, Any],
        extra_parameters: Mapping[str, nn.Parameter] | None = None,
    ) -> None:
        parameters = dict(model.named_parameters())
        parameters.update(extra_parameters or {})
        optimizer.state.clear()
        for name, value in saved.get("state", {}).items():
            parameter = parameters.get(str(name))
            if parameter is None:
                continue
            optimizer.state[parameter] = {
                key: (
                    item.cpu()
                    if key == "step" and isinstance(item, torch.Tensor)
                    else StreamingForecastingTrainer._move_optimizer_value(
                        item,
                        parameter.device,
                    )
                )
                for key, item in value.items()
            }
        for group, options in zip(
            optimizer.param_groups,
            saved.get("param_group_options", []),
        ):
            group.update({key: value for key, value in options.items() if key != "params"})

    @staticmethod
    def _copy_optimizer_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {
                key: StreamingForecastingTrainer._copy_optimizer_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [StreamingForecastingTrainer._copy_optimizer_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(StreamingForecastingTrainer._copy_optimizer_value(item) for item in value)
        return value

    @staticmethod
    def _move_optimizer_value(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {
                key: StreamingForecastingTrainer._move_optimizer_value(item, device)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [StreamingForecastingTrainer._move_optimizer_value(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(
                StreamingForecastingTrainer._move_optimizer_value(item, device) for item in value
            )
        return value

    def _prepare_model_for_dataset(
            self,
            dataset: Dataset,
            *,
            model: nn.Module | None = None,
    ) -> None:
        """
        Prepare a model for the temporal and spatial shape of a dataset.

        In the global continual protocol, different traffic datasets may
        have different input lengths, forecasting horizons, and numbers
        of nodes. Only the lightweight I/O adapters are dataset-shape
        dependent. The shared Transformer and the hierarchical
        Domain/State Expert banks must remain persistent.
        """
        target_model = self.model if model is None else model

        if len(dataset) <= 0:
            raise ValueError(
                "cannot prepare model for an empty dataset"
            )

        # --------------------------------------------------------
        # Infer the actual temporal/spatial shape from one sample.
        #
        # x: [T, N, F]
        # y: [H, N]
        # --------------------------------------------------------
        sample_x, sample_y, _ = unpack_sample(dataset[0])

        if sample_x.ndim != 3:
            raise ValueError(
                "expected dataset input sample with shape "
                f"[T,N,F], got {tuple(sample_x.shape)}"
            )

        if sample_y.ndim < 2:
            raise ValueError(
                "expected dataset target sample with shape "
                f"[H,N], got {tuple(sample_y.shape)}"
            )

        input_length = int(sample_x.shape[0])
        num_nodes = int(sample_x.shape[1])
        horizon = int(sample_y.shape[0])

        # --------------------------------------------------------
        # Configure the temporal I/O adapters.
        #
        # IMPORTANT:
        # This must NOT rebuild/reset the shared Transformer,
        # Domain Router, Domain Experts, State Router, State Experts,
        # or their prototypes.
        # --------------------------------------------------------
        if hasattr(target_model, "configure_io_shape"):
            target_model.configure_io_shape(
                input_length=input_length,
                horizon=horizon,
            )

        # --------------------------------------------------------
        # Preserve the existing node-handling behaviour.
        # --------------------------------------------------------
        node_keys = getattr(dataset, "node_keys", None)

        if (
                node_keys is not None
                and hasattr(target_model, "configure_node_ids")
        ):
            target_model.configure_node_ids(node_keys)

        elif hasattr(target_model, "expand_node_params"):
            target_model.expand_node_params(num_nodes)

        # Dynamic modules created above must live on the same device.
        target_model.to(self.device)

    def _adjacency_tensor(self, dataset: Dataset) -> torch.Tensor | None:
        adjacency = getattr(dataset, "adjacency", None)
        if adjacency is None:
            return None
        return torch.as_tensor(adjacency, dtype=torch.float32, device=self.device)

    def _model_forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        try:
            return self.model(x, adj=adj)
        except TypeError:
            return self.model(x)


def target_array(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_mode: str,
    flow_feature_index: int,
) -> np.ndarray:
    if target_mode == "residual":
        return y - np.repeat(x[:, -1, :, flow_feature_index][:, None, :], y.shape[1], axis=1)
    if target_mode == "direct":
        return y
    raise ValueError(f"unsupported target mode: {target_mode}")


def last_value_forecast(x: torch.Tensor, flow_feature_index: int, horizon: int) -> torch.Tensor:
    last_value = x[:, -1, :, flow_feature_index]
    return last_value[:, None, :].repeat(1, horizon, 1)


def unpack_sample(sample):
    if len(sample) == 2:
        x, y = sample
        return x, y, np.ones_like(y, dtype=np.float32)
    if len(sample) == 3:
        return sample
    raise ValueError(f"dataset sample must contain 2 or 3 items, got {len(sample)}")


def unpack_batch(batch):
    if len(batch) == 2:
        x, y = batch
        return x, y, torch.ones_like(y, dtype=torch.bool)
    if len(batch) == 3:
        return batch
    raise ValueError(f"dataloader batch must contain 2 or 3 items, got {len(batch)}")
