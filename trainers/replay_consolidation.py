"""Optional final consolidation using stored TRAIN replay and historical VAL only."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from utils.checkpointing import atomic_torch_save, try_load_torch, capture_rng_state, restore_rng_state


def consolidate_replay(trainer, bundles, routes, *, epochs, output_dir):
    if trainer.method != "expert_replay_distill" or not trainer.memory.entries:
        raise ValueError("Consolidation requires the existing expert replay memory")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    best_path, last_path = directory / "consolidation_best.pt", directory / "consolidation_last.pt"
    replay_datasets = {b["period"].name: b["train"] for b in bundles}
    steps = max(math.ceil(len(b["train"]) / int(trainer.config.get("batch_size", 64))) for b in bundles)
    parameters = list(trainer.model.parameters())
    if trainer._use_teacher_gate():
        parameters.extend(trainer.teacher_gate.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(trainer.config.get("lr", 1e-3)),
                                 weight_decay=float(trainer.config.get("weight_decay", 1e-4)))

    def save_state():
        return {"model": trainer.model.state_dict(), "continual": trainer.continual_state_dict(),
                "rng": capture_rng_state()}

    def restore(state):
        trainer.model.load_state_dict(state["model"], strict=True)
        trainer.load_continual_state(state["continual"], replay_datasets)
        restore_rng_state(state["rng"])

    def validate():
        # Validation must not change which TRAIN replay samples are drawn next.
        rng = capture_rng_state()
        by_dataset = {}
        try:
            for bundle, route in zip(bundles, routes):
                result = trainer.evaluate(bundle["val"], task_name=bundle["period"].name,
                                          domain_id=route["domain_id"])
                if not np.isfinite(result["relative_mae"]):
                    raise ValueError("Validation Skill is undefined; consolidation cannot rank this stream")
                by_dataset.setdefault(bundle["dataset"], []).append(float(result["relative_mae"]))
        finally:
            restore_rng_state(rng)
        means = {name: float(np.mean(values)) for name, values in by_dataset.items()}
        # One weak dataset must not be hidden by averaging the eight easier ones.
        rank = (max(means.values()), float(np.mean(list(means.values()))))
        return rank, {name: 1.0 - value for name, value in means.items()}

    history, start, best_rank, best_epoch = [], 1, None, 0
    last = try_load_torch(last_path, map_location=trainer.device)
    if last is not None:
        restore(last)
        trainer.base._load_optimizer_state_by_name(optimizer, trainer.model, last["optimizer"],
            extra_parameters={"teacher_gate." + n: p for n, p in trainer.teacher_gate.named_parameters()})
        history, start = last["history"], int(last["next_epoch"])
        best_rank, best_epoch = tuple(last["best_rank"]), int(last["best_epoch"])
    else:
        best_rank, scores = validate()
        history.append({"epoch": 0, "rank": best_rank, "validation_skill": scores})
        atomic_torch_save(save_state(), best_path)

    for epoch in range(start, epochs + 1):
        losses = []
        trainer.model.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            losses.append(trainer._backward_balanced_expert_replay(optimizer, current_loss=None))
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), float(trainer.config.get("grad_clip", 1.0)))
            optimizer.step()
            if trainer._use_ema_teacher():
                trainer._update_ema_teacher()
        optimizer.zero_grad(set_to_none=True)
        trainer._refresh_domain_prototypes_from_memory()
        trainer._refresh_state_prototypes_from_memory()
        rank, scores = validate()
        history.append({"epoch": epoch, "train_replay_loss": float(np.mean(losses)),
                        "rank": rank, "validation_skill": scores})
        if rank < best_rank:
            best_rank, best_epoch = rank, epoch
            atomic_torch_save(save_state(), best_path)
        state = save_state()
        state.update(next_epoch=epoch + 1, history=history, best_rank=best_rank, best_epoch=best_epoch,
            optimizer=trainer.base._optimizer_state_by_name(optimizer, trainer.model,
                extra_parameters={"teacher_gate." + n: p for n, p in trainer.teacher_gate.named_parameters()}))
        atomic_torch_save(state, last_path)
        (directory / "consolidation_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"CONSOLIDATE epoch={epoch}/{epochs} steps={steps} worst_val_skill={1-rank[0]:.6f} best_epoch={best_epoch}", flush=True)
    restore(try_load_torch(best_path, map_location=trainer.device))
    return {"epochs": epochs, "steps_per_epoch": steps, "selected_epoch": best_epoch,
            "selected_validation_rank": best_rank, "history": history,
            "training_source": "historical TRAIN replay only", "selection_source": "historical VAL only"}
