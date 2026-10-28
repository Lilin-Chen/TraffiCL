from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from models.backbones.lightweight_transformer import LightweightTrafficTransformer, LiteInvertedTrafficTransformer, LiteCrossNodeBlock


CREATE_DOMAIN = "CREATE_DOMAIN"
REUSE_DOMAIN = "REUSE_DOMAIN"
CREATE_STATE = "CREATE_STATE"
REUSE_STATE = "REUSE_STATE"


@dataclass(frozen=True)
class RoutingDecision:
    action: str
    selected_id: int
    similarities: tuple[float, ...]
    max_similarity: float | None


class FeedForwardAdapter(nn.Module):
    """Zero-initialized bottleneck adapter used by canonical TraffiCL experts."""

    def __init__(self, d_model: int, bottleneck: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, d_model),
        )
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureGroupEncoder(nn.Module):
    """Shared input encoder for available feature groups; it is not a regime router."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        feature_groups: Mapping[str, Sequence[int]],
        dropout: float,
    ):
        super().__init__()
        normalized = {
            str(name): [int(index) for index in indices]
            for name, indices in feature_groups.items()
            if indices
        }
        if not normalized:
            raise ValueError("feature_groups must contain at least one non-empty group")
        for name, indices in normalized.items():
            if min(indices) < 0 or max(indices) >= input_dim:
                raise ValueError(f"feature group {name!r} contains an invalid column index")

        self.group_names = list(normalized)
        self.feature_groups = normalized
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(len(indices), d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                )
                for name, indices in normalized.items()
            }
        )
        self.mixing_weights = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(self.group_names)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = x.mean(dim=(1, 2))
        weights = torch.softmax(self.mixing_weights(context), dim=-1)
        outputs = [
            self.projections[name](x[..., self.feature_groups[name]])
            for name in self.group_names
        ]
        stacked = torch.stack(outputs, dim=1)
        return torch.sum(stacked * weights[:, :, None, None, None], dim=1)


class TrafficDomainExpertBank(nn.Module):
    """Create, store, and execute domain adapters selected by the router."""

    def __init__(self, d_model: int, bottleneck: int, dropout: float):
        super().__init__()
        self.d_model = int(d_model)
        self.bottleneck = int(bottleneck)
        self.dropout = float(dropout)
        self.experts = nn.ModuleList()
        self.gate_logits = nn.ParameterList()

    def __len__(self) -> int:
        return len(self.experts)

    def create_expert(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> int:
        expert = FeedForwardAdapter(self.d_model, self.bottleneck, self.dropout)
        if device is not None:
            expert = expert.to(device=device, dtype=dtype)
        self.experts.append(expert)
        self.gate_logits.append(
            nn.Parameter(torch.tensor(-2.0, device=device, dtype=dtype or torch.float32))
        )
        return len(self.experts) - 1

    def rebuild(
        self,
        count: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.experts = nn.ModuleList()
        self.gate_logits = nn.ParameterList()
        for _ in range(int(count)):
            self.create_expert(device=device, dtype=dtype)

    def forward(self, hidden: torch.Tensor, domain_expert_id: int) -> torch.Tensor:
        expert_id = int(domain_expert_id)
        if expert_id < 0 or expert_id >= len(self.experts):
            raise IndexError(f"domain expert id is unavailable: {expert_id}")
        gate = torch.sigmoid(self.gate_logits[expert_id]).to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return hidden + gate * self.experts[expert_id](hidden)

    def gate_value(self, domain_expert_id: int) -> torch.Tensor:
        expert_id = int(domain_expert_id)
        if expert_id < 0 or expert_id >= len(self.gate_logits):
            raise IndexError(f"domain expert gate id is unavailable: {expert_id}")
        return torch.sigmoid(self.gate_logits[expert_id])


class TrafficStateExpertBank(nn.Module):
    """Create, store, and execute state adapters for one traffic domain."""

    def __init__(self, d_model: int, bottleneck: int, dropout: float):
        super().__init__()
        self.d_model = int(d_model)
        self.bottleneck = int(bottleneck)
        self.dropout = float(dropout)
        self.experts = nn.ModuleList()
        self.gate_logits = nn.ParameterList()

    def __len__(self) -> int:
        return len(self.experts)

    def create_expert(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> int:
        expert = FeedForwardAdapter(self.d_model, self.bottleneck, self.dropout)
        if device is not None:
            expert = expert.to(device=device, dtype=dtype)
        self.experts.append(expert)
        self.gate_logits.append(
            nn.Parameter(torch.tensor(-2.0, device=device, dtype=dtype or torch.float32))
        )
        return len(self.experts) - 1

    def rebuild(
        self,
        count: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.experts = nn.ModuleList()
        self.gate_logits = nn.ParameterList()
        for _ in range(int(count)):
            self.create_expert(device=device, dtype=dtype)

    def forward(self, hidden: torch.Tensor, state_expert_id: int) -> torch.Tensor:
        expert_id = int(state_expert_id)
        if expert_id < 0 or expert_id >= len(self.experts):
            raise IndexError(f"state expert id is unavailable: {expert_id}")
        gate = torch.sigmoid(self.gate_logits[expert_id]).to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return hidden + gate * self.experts[expert_id](hidden)

    def gate_value(self, state_expert_id: int) -> torch.Tensor:
        expert_id = int(state_expert_id)
        if expert_id < 0 or expert_id >= len(self.gate_logits):
            raise IndexError(f"state expert gate id is unavailable: {expert_id}")
        return torch.sigmoid(self.gate_logits[expert_id])


class HierarchicalRegimeRouter(nn.Module):
    """The sole Domain -> State expert selector and prototype memory."""

    def __init__(
        self,
        *,
        d_model: int,
        dynamic_domain_experts: bool,
        dynamic_state_experts: bool,
        max_domain_experts: int,
        max_state_experts_per_domain: int,
        domain_similarity_threshold: float,
        state_similarity_threshold: float,
        domain_prototype_update_rate: float,
        state_prototype_update_rate: float,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.dynamic_domain_experts = bool(dynamic_domain_experts)
        self.dynamic_state_experts = bool(dynamic_state_experts)
        self.max_domain_experts = int(max_domain_experts)
        self.max_state_experts_per_domain = int(max_state_experts_per_domain)
        self.domain_similarity_threshold = float(domain_similarity_threshold)
        self.state_similarity_threshold = float(state_similarity_threshold)
        self.domain_prototype_update_rate = float(domain_prototype_update_rate)
        self.state_prototype_update_rate = float(state_prototype_update_rate)
        self._validate_config()

        self.domain_prototypes: list[torch.Tensor] = []
        self.state_prototypes: list[list[torch.Tensor]] = []
        self.domain_usage_counts: list[int] = []
        self.state_usage_counts: list[list[int]] = []
        self.task_domain_assignment_history: list[dict[str, Any]] = []
        self.active_domain_id: int | None = None
        self.active_task_id: str | None = None
        self.current_task_state_usage: Counter[int] = Counter()
        self.current_task_state_actions: Counter[str] = Counter()
        self.current_task_state_similarity_sum = 0.0
        self.current_task_state_similarity_count = 0
        self.current_task_state_similarity_min: float | None = None
        self.current_task_state_similarity_max: float | None = None

    @staticmethod
    def pool_windows(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"expected hidden [B,N,D], got shape {tuple(hidden.shape)}")
        return hidden.mean(dim=1)

    def route_domain(
        self,
        descriptor: torch.Tensor,
        *,
        allow_create: bool,
    ) -> RoutingDecision:
        query = self._normalize_descriptor(descriptor)
        if not self.domain_prototypes:
            if not allow_create:
                raise RuntimeError("domain routing requires at least one existing domain expert")
            return RoutingDecision(CREATE_DOMAIN, 0, (), None)

        similarities = self._similarities(query, self.domain_prototypes)
        nearest_id = int(torch.tensor(similarities).argmax().item())
        maximum = similarities[nearest_id]
        can_create = (
            allow_create
            and self.dynamic_domain_experts
            and len(self.domain_prototypes) < self.max_domain_experts
        )
        if maximum < self.domain_similarity_threshold and can_create:
            return RoutingDecision(
                CREATE_DOMAIN,
                len(self.domain_prototypes),
                similarities,
                maximum,
            )
        return RoutingDecision(REUSE_DOMAIN, nearest_id, similarities, maximum)

    def commit_domain(
        self,
        decision: RoutingDecision,
        descriptor: torch.Tensor,
        *,
        task_id: str,
        update_prototype: bool,
        record_assignment: bool,
    ) -> None:
        value = self._normalize_descriptor(descriptor)
        domain_id = int(decision.selected_id)
        if decision.action == CREATE_DOMAIN:
            if domain_id != len(self.domain_prototypes):
                raise RuntimeError("new domain id must append to the prototype memory")
            self.domain_prototypes.append(value)
            self.state_prototypes.append([])
            self.domain_usage_counts.append(0)
            self.state_usage_counts.append([])
        elif decision.action == REUSE_DOMAIN:
            self._require_domain(domain_id)
            if update_prototype:
                self.domain_prototypes[domain_id] = self._ema_update(
                    self.domain_prototypes[domain_id],
                    value,
                    self.domain_prototype_update_rate,
                )
        else:
            raise ValueError(f"unsupported domain action: {decision.action}")

        if record_assignment:
            self.domain_usage_counts[domain_id] += 1
        self.active_domain_id = domain_id
        self.active_task_id = str(task_id)
        self.current_task_state_usage = Counter()
        self.current_task_state_actions = Counter()
        self.current_task_state_similarity_sum = 0.0
        self.current_task_state_similarity_count = 0
        self.current_task_state_similarity_min = None
        self.current_task_state_similarity_max = None
        if record_assignment:
            self.task_domain_assignment_history.append(
                {
                    "task_id": str(task_id),
                    "domain_id": domain_id,
                    "action": decision.action,
                    "similarities": list(decision.similarities),
                    "max_similarity": decision.max_similarity,
                }
            )

    def route_state(
        self,
        descriptor: torch.Tensor,
        domain_id: int,
        *,
        allow_create: bool,
    ) -> RoutingDecision:
        domain_id = int(domain_id)
        self._require_domain(domain_id)
        query = self._normalize_descriptor(descriptor)
        prototypes = self.state_prototypes[domain_id]
        if not prototypes:
            if not allow_create:
                raise RuntimeError(
                    f"domain {domain_id} has no state expert available for inference"
                )
            return RoutingDecision(CREATE_STATE, 0, (), None)

        similarities = self._similarities(query, prototypes)
        nearest_id = int(torch.tensor(similarities).argmax().item())
        maximum = similarities[nearest_id]
        can_create = (
            allow_create
            and self.dynamic_state_experts
            and len(prototypes) < self.max_state_experts_per_domain
        )
        if maximum < self.state_similarity_threshold and can_create:
            return RoutingDecision(CREATE_STATE, len(prototypes), similarities, maximum)
        return RoutingDecision(REUSE_STATE, nearest_id, similarities, maximum)

    def commit_state(
        self,
        decision: RoutingDecision,
        descriptor: torch.Tensor,
        domain_id: int,
        *,
        update_prototype: bool,
        record_usage: bool,
    ) -> None:
        domain_id = int(domain_id)
        self._require_domain(domain_id)
        state_id = int(decision.selected_id)
        value = self._normalize_descriptor(descriptor)
        prototypes = self.state_prototypes[domain_id]
        counts = self.state_usage_counts[domain_id]
        if decision.action == CREATE_STATE:
            if state_id != len(prototypes):
                raise RuntimeError("new state id must append within its domain")
            prototypes.append(value)
            counts.append(0)
        elif decision.action == REUSE_STATE:
            if state_id < 0 or state_id >= len(prototypes):
                raise IndexError(f"state prototype is unavailable: domain={domain_id} state={state_id}")
            if update_prototype:
                prototypes[state_id] = self._ema_update(
                    prototypes[state_id],
                    value,
                    self.state_prototype_update_rate,
                )
        else:
            raise ValueError(f"unsupported state action: {decision.action}")

        if record_usage:
            counts[state_id] += 1
            self.current_task_state_usage[state_id] += 1
            self.current_task_state_actions[decision.action] += 1
            if decision.max_similarity is not None:
                similarity = float(decision.max_similarity)
                self.current_task_state_similarity_sum += similarity
                self.current_task_state_similarity_count += 1
                self.current_task_state_similarity_min = (
                    similarity
                    if self.current_task_state_similarity_min is None
                    else min(self.current_task_state_similarity_min, similarity)
                )
                self.current_task_state_similarity_max = (
                    similarity
                    if self.current_task_state_similarity_max is None
                    else max(self.current_task_state_similarity_max, similarity)
                )

    def recorded_domain_for_task(
            self,
            task_id: str,
    ) -> int | None:
        """
        Return the historical Domain assignment for a previously
        observed task without rerouting it through current prototypes.
        """
        task_id = str(task_id)

        for item in reversed(
                self.task_domain_assignment_history
        ):
            if str(item.get("task_id")) != task_id:
                continue

            domain_id = int(item["domain_id"])
            self._require_domain(domain_id)
            return domain_id

        return None

    def activate_recorded_domain(self, domain_id: int, *, task_id: str | None = None) -> None:
        self._require_domain(int(domain_id))
        self.active_domain_id = int(domain_id)
        if task_id is not None:
            self.active_task_id = str(task_id)

    def state_usage_summary(self, domain_id: int | None = None) -> dict[str, Any]:
        selected = self.active_domain_id if domain_id is None else int(domain_id)
        if selected is None:
            return {"domain_id": None, "num_state_experts": 0, "usage": {}}
        self._require_domain(selected)
        total = sum(self.current_task_state_usage.values())
        usage = {
            str(state_id): {
                "count": int(count),
                "ratio": float(count / total) if total else 0.0,
            }
            for state_id, count in sorted(self.current_task_state_usage.items())
        }
        return {
            "domain_id": selected,
            "num_state_experts": len(self.state_prototypes[selected]),
            "usage": usage,
            "actions": dict(self.current_task_state_actions),
            "mean_max_similarity": (
                self.current_task_state_similarity_sum
                / self.current_task_state_similarity_count
                if self.current_task_state_similarity_count
                else None
            ),
            "min_max_similarity": self.current_task_state_similarity_min,
            "max_max_similarity": self.current_task_state_similarity_max,
        }

    def export_state(self) -> dict[str, Any]:
        domain_memory = []
        for domain_id, prototype in enumerate(self.domain_prototypes):
            states = [
                {
                    "domain_id": domain_id,
                    "state_id": state_id,
                    "prototype": state_prototype.detach().cpu().clone(),
                    "usage_count": int(self.state_usage_counts[domain_id][state_id]),
                }
                for state_id, state_prototype in enumerate(self.state_prototypes[domain_id])
            ]
            domain_memory.append(
                {
                    "domain_id": domain_id,
                    "prototype": prototype.detach().cpu().clone(),
                    "usage_count": int(self.domain_usage_counts[domain_id]),
                    "states": states,
                }
            )
        return {
            "config": self.config_dict(),
            "num_domain_experts": len(domain_memory),
            "num_state_experts": [len(item["states"]) for item in domain_memory],
            "domain_memory": domain_memory,
            "task_domain_assignment_history": list(self.task_domain_assignment_history),
            "active_domain_id": self.active_domain_id,
            "active_task_id": self.active_task_id,
            "current_task_state_usage": dict(self.current_task_state_usage),
            "current_task_state_actions": dict(self.current_task_state_actions),
            "current_task_state_similarity_sum": self.current_task_state_similarity_sum,
            "current_task_state_similarity_count": self.current_task_state_similarity_count,
            "current_task_state_similarity_min": self.current_task_state_similarity_min,
            "current_task_state_similarity_max": self.current_task_state_similarity_max,
        }

    def load_exported_state(self, state: Mapping[str, Any]) -> None:
        saved_config = dict(state.get("config", {}))
        for name in self.config_dict():
            if name in saved_config:
                setattr(self, name, saved_config[name])
        self._validate_config()

        memory = list(state.get("domain_memory", []))
        self.domain_prototypes = []
        self.state_prototypes = []
        self.domain_usage_counts = []
        self.state_usage_counts = []
        for expected_domain_id, item in enumerate(memory):
            domain_id = int(item["domain_id"])
            if domain_id != expected_domain_id:
                raise ValueError("checkpoint domain ids must be contiguous and ordered")
            self.domain_prototypes.append(self._normalize_descriptor(item["prototype"]))
            self.domain_usage_counts.append(int(item.get("usage_count", 0)))
            domain_states: list[torch.Tensor] = []
            domain_counts: list[int] = []
            for expected_state_id, state_item in enumerate(item.get("states", [])):
                if int(state_item["domain_id"]) != domain_id:
                    raise ValueError("checkpoint state prototype has the wrong domain id")
                if int(state_item["state_id"]) != expected_state_id:
                    raise ValueError("checkpoint state ids must be contiguous and ordered")
                domain_states.append(self._normalize_descriptor(state_item["prototype"]))
                domain_counts.append(int(state_item.get("usage_count", 0)))
            self.state_prototypes.append(domain_states)
            self.state_usage_counts.append(domain_counts)

        if len(self.domain_prototypes) > self.max_domain_experts:
            raise ValueError("checkpoint contains more domain experts than the configured maximum")
        if any(
            len(prototypes) > self.max_state_experts_per_domain
            for prototypes in self.state_prototypes
        ):
            raise ValueError("checkpoint contains more state experts than the configured maximum")

        self.task_domain_assignment_history = [
            dict(item) for item in state.get("task_domain_assignment_history", [])
        ]
        active = state.get("active_domain_id")
        self.active_domain_id = None if active is None else int(active)
        self.active_task_id = (
            None if state.get("active_task_id") is None else str(state["active_task_id"])
        )
        self.current_task_state_usage = Counter(
            {int(key): int(value) for key, value in state.get("current_task_state_usage", {}).items()}
        )
        self.current_task_state_actions = Counter(
            {
                str(key): int(value)
                for key, value in state.get("current_task_state_actions", {}).items()
            }
        )
        self.current_task_state_similarity_sum = float(
            state.get("current_task_state_similarity_sum", 0.0)
        )
        self.current_task_state_similarity_count = int(
            state.get("current_task_state_similarity_count", 0)
        )
        similarity_min = state.get("current_task_state_similarity_min")
        similarity_max = state.get("current_task_state_similarity_max")
        self.current_task_state_similarity_min = (
            None if similarity_min is None else float(similarity_min)
        )
        self.current_task_state_similarity_max = (
            None if similarity_max is None else float(similarity_max)
        )
        if self.active_domain_id is not None:
            self._require_domain(self.active_domain_id)

    def config_dict(self) -> dict[str, Any]:
        return {
            "dynamic_domain_experts": self.dynamic_domain_experts,
            "dynamic_state_experts": self.dynamic_state_experts,
            "max_domain_experts": self.max_domain_experts,
            "max_state_experts_per_domain": self.max_state_experts_per_domain,
            "domain_similarity_threshold": self.domain_similarity_threshold,
            "state_similarity_threshold": self.state_similarity_threshold,
            "domain_prototype_update_rate": self.domain_prototype_update_rate,
            "state_prototype_update_rate": self.state_prototype_update_rate,
        }

    def _normalize_descriptor(self, descriptor: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(descriptor).detach().float().cpu().reshape(-1)
        if value.numel() != self.d_model:
            raise ValueError(
                f"expected descriptor dimension {self.d_model}, got {value.numel()}"
            )
        return F.normalize(value, dim=0, eps=1e-12)

    @staticmethod
    def _similarities(query: torch.Tensor, prototypes: Sequence[torch.Tensor]) -> tuple[float, ...]:
        matrix = torch.stack([prototype.float().cpu() for prototype in prototypes], dim=0)
        values = F.cosine_similarity(query.unsqueeze(0), matrix, dim=-1)
        return tuple(float(value) for value in values.tolist())

    def _ema_update(
        self,
        old: torch.Tensor,
        new: torch.Tensor,
        rate: float,
    ) -> torch.Tensor:
        return self._normalize_descriptor((1.0 - float(rate)) * old + float(rate) * new)

    def _require_domain(self, domain_id: int) -> None:
        if domain_id < 0 or domain_id >= len(self.domain_prototypes):
            raise IndexError(f"domain prototype is unavailable: {domain_id}")

    def _validate_config(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.max_domain_experts <= 0:
            raise ValueError("max_domain_experts must be positive")
        if self.max_state_experts_per_domain <= 0:
            raise ValueError("max_state_experts_per_domain must be positive")
        for name in ("domain_similarity_threshold", "state_similarity_threshold"):
            value = float(getattr(self, name))
            if value < -1.0 or value > 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
        for name in ("domain_prototype_update_rate", "state_prototype_update_rate"):
            value = float(getattr(self, name))
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


class ExpertLightweightTrafficTransformer(LightweightTrafficTransformer):
    """TraCL backbone with sequential, representation-driven Domain -> State routing."""

    def __init__(
        self,
        *,
        input_dim: int,
        horizon: int,
        num_series: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_series: int | None = None,
        use_spatial_mixer: bool = True,
        initial_node_ids: Sequence[str] | None = None,
        adapter_bottleneck: int = 16,
        feature_groups: Mapping[str, Sequence[int]] | None = None,
        dynamic_domain_experts: bool = True,
        dynamic_state_experts: bool = True,
        max_domain_experts: int = 8,
        max_state_experts_per_domain: int = 8,
        domain_similarity_threshold: float = 0.85,
        state_similarity_threshold: float = 0.85,
        domain_prototype_update_rate: float = 0.1,
        state_prototype_update_rate: float = 0.1,
        domain_descriptor_max_windows: int = 2048,
    ):
        super().__init__(
            input_dim=input_dim,
            horizon=horizon,
            num_series=num_series,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_series=max_series,
            use_spatial_mixer=use_spatial_mixer,
            initial_node_ids=initial_node_ids,
        )
        configured_groups = feature_groups or {
            "mobility": [0],
            "time": list(range(1, input_dim)),
        }
        groups = {
            str(name): [int(index) for index in indices if 0 <= int(index) < input_dim]
            for name, indices in configured_groups.items()
        }
        groups = {name: indices for name, indices in groups.items() if indices}
        if not groups:
            groups = {"mobility": [0]}

        self.d_model = int(d_model)
        self.adapter_bottleneck = int(adapter_bottleneck)
        self.adapter_dropout = float(dropout)
        self.domain_descriptor_max_windows = int(domain_descriptor_max_windows)
        if self.domain_descriptor_max_windows <= 0:
            raise ValueError("domain_descriptor_max_windows must be positive")
        self.feature_group_encoder = FeatureGroupEncoder(
            input_dim=input_dim,
            d_model=d_model,
            feature_groups=groups,
            dropout=dropout,
        )
        self.domain_experts = TrafficDomainExpertBank(
            d_model=d_model,
            bottleneck=adapter_bottleneck,
            dropout=dropout,
        )
        self.state_expert_banks = nn.ModuleList()
        self.regime_router = HierarchicalRegimeRouter(
            d_model=d_model,
            dynamic_domain_experts=dynamic_domain_experts,
            dynamic_state_experts=dynamic_state_experts,
            max_domain_experts=max_domain_experts,
            max_state_experts_per_domain=max_state_experts_per_domain,
            domain_similarity_threshold=domain_similarity_threshold,
            state_similarity_threshold=state_similarity_threshold,
            domain_prototype_update_rate=domain_prototype_update_rate,
            state_prototype_update_rate=state_prototype_update_rate,
        )
        self.routing_updates_enabled = True
        self.horizon_gate_logits = nn.Parameter(torch.full((self.horizon,), -2.0))
        gate_hidden = max(4, self.d_model // 2)
        self.sample_gate = nn.Sequential(
            nn.Linear(self.d_model, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.horizon),
        )
        nn.init.zeros_(self.sample_gate[-1].weight)
        nn.init.zeros_(self.sample_gate[-1].bias)
        if isinstance(self.head, nn.Sequential) and isinstance(self.head[-1], nn.Linear):
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        self.register_buffer(
            "residual_zero_normalized",
            torch.zeros(1, self.horizon, 1),
            persistent=False,
        )
        self.last_routing_trace: dict[str, Any] = {}
        self._forced_state_ids: list[int] | None = None
    @property
    def active_domain_id(self) -> int | None:
        return self.regime_router.active_domain_id

    @property
    def active_task_id(self) -> str | None:
        return self.regime_router.active_task_id

    def _project_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return super()._project_inputs(x) + self.feature_group_encoder(x)

    def domain_descriptor_samples(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized task samples h_i = Pool(H_i), shape [B, D]."""

        return self.regime_router.pool_windows(self.encode(x))

    def route_task_descriptor(
        self,
        descriptor: torch.Tensor,
        *,
        task_id: str,
        allow_create: bool,
        update_prototype: bool,
        record_assignment: bool = True,
        log: bool = True,
    ) -> RoutingDecision:
        if len(self.domain_experts) != len(self.regime_router.domain_prototypes):
            raise RuntimeError("domain expert bank and router prototype memory are out of sync")
        decision = self.regime_router.route_domain(descriptor, allow_create=allow_create)
        if decision.action == CREATE_DOMAIN:
            created_id = self._create_domain_expert()
            if created_id != decision.selected_id:
                raise RuntimeError("router and domain bank creation ids diverged")
        self.regime_router.commit_domain(
            decision,
            descriptor,
            task_id=task_id,
            update_prototype=update_prototype,
            record_assignment=record_assignment,
        )
        if log:
            print(
                "[Domain Routing] "
                f"task_id={task_id} num_domain_experts={len(self.domain_experts)} "
                f"domain_similarities={list(decision.similarities)} "
                f"selected_domain_id={decision.selected_id} "
                f"domain_action={decision.action} "
                f"domain_max_similarity={decision.max_similarity}",
                flush=True,
            )
        return decision

    def recorded_domain_for_task(
            self,
            task_id: str,
    ) -> int | None:
        return self.regime_router.recorded_domain_for_task(
            task_id
        )

    def activate_recorded_domain(self, domain_id: int, *, task_id: str | None = None) -> None:
        self.regime_router.activate_recorded_domain(domain_id, task_id=task_id)

    def set_routing_updates(self, enabled: bool) -> None:
        self.routing_updates_enabled = bool(enabled)

    def set_forced_state_ids(self, state_ids: Sequence[int] | None) -> None:
        """Temporarily force replay samples to use their recorded State Experts."""
        if state_ids is None:
            self._forced_state_ids = None
            return

        self._forced_state_ids = [int(value) for value in state_ids]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecast, _ = self.forward_with_components(x)
        return forecast

    def forward_with_components(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[int] | int]]:
        domain_id = self.active_domain_id
        if domain_id is None:
            raise RuntimeError(
                "no active traffic domain; compute one task-level descriptor and route it first"
            )
        if domain_id >= len(self.domain_experts):
            raise RuntimeError("active domain is missing from TrafficDomainExpertBank")

        backbone_hidden = self.encode(x)
        pooled_hidden = backbone_hidden.mean(dim=1)
        sample_gate_logits = self.sample_gate(pooled_hidden)
        horizon_prior = self.horizon_gate_logits.to(device=x.device, dtype=x.dtype).view(1, -1)
        safety_gate = torch.sigmoid(horizon_prior + sample_gate_logits).unsqueeze(-1)
        zero_offset = self.residual_zero_normalized.to(device=x.device, dtype=x.dtype)
        backbone_residual_raw = self.forecast_from_hidden(backbone_hidden)
        backbone_forecast = zero_offset + safety_gate * backbone_residual_raw
        domain_hidden = self.domain_experts(backbone_hidden, domain_id)
        state_hidden, state_ids = self._apply_state_experts(domain_hidden, domain_id)
        adapted_residual_raw = self.forecast_from_hidden(state_hidden)
        adapted_forecast = zero_offset + safety_gate * adapted_residual_raw
        expert_contribution = adapted_forecast - backbone_forecast
        final_forecast = adapted_forecast
        domain_gate = self.domain_experts.gate_value(domain_id).to(device=x.device, dtype=x.dtype)
        state_gate_values = [
            self.state_expert_banks[domain_id].gate_value(state_id).to(device=x.device, dtype=x.dtype)
            for state_id in state_ids
        ]
        state_gate = torch.stack(state_gate_values).mean() if state_gate_values else torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
        self.last_routing_trace = {
            "backbone_hidden_shape": tuple(backbone_hidden.shape),
            "domain_hidden_shape": tuple(domain_hidden.shape),
            "state_hidden_shape": tuple(state_hidden.shape),
            "state_descriptor_source": "domain_hidden",
            "domain_id": domain_id,
            "state_ids": list(state_ids),
            "mean_safety_gate": float(safety_gate.detach().mean().cpu()),
            "mean_sample_gate": float(torch.sigmoid(sample_gate_logits.detach()).mean().cpu()),
            "domain_gate": float(domain_gate.detach().cpu()),
            "state_gate": float(state_gate.detach().cpu()),
        }
        return final_forecast, {
            "backbone_hidden": backbone_hidden,
            "backbone_forecast": backbone_forecast,
            "backbone_residual": backbone_residual_raw,
            "safety_gate": safety_gate,
            "sample_gate": torch.sigmoid(sample_gate_logits),
            "horizon_gate": torch.sigmoid(horizon_prior).view(-1),
            "domain_hidden": domain_hidden,
            "state_hidden": state_hidden,
            "state_ids": state_ids,
            "domain_id": domain_id,
            "expert_contribution": expert_contribution,
            "domain_gate": domain_gate,
            "state_gate": state_gate,
        }

    def routing_summary(self) -> dict[str, Any]:
        return {
            "num_domain_experts": len(self.domain_experts),
            "num_state_experts": [len(bank) for bank in self.state_expert_banks],
            "active_task_id": self.active_task_id,
            "active_domain_id": self.active_domain_id,
            "state_routing": self.regime_router.state_usage_summary(),
            "horizon_gates": self.horizon_gate_values(),
            "domain_gates": self.domain_gate_values(),
            "state_gates": self.state_gate_values(),
        }

    def route_state_descriptors(
        self,
        descriptors: torch.Tensor,
        domain_id: int,
        *,
        allow_create: bool,
        update_prototype: bool,
        record_usage: bool = True,
    ) -> list[RoutingDecision]:
        if descriptors.ndim != 2 or descriptors.shape[1] != self.d_model:
            raise ValueError(
                f"expected state descriptors [B,{self.d_model}], got {tuple(descriptors.shape)}"
            )
        domain_id = int(domain_id)
        if domain_id < 0 or domain_id >= len(self.state_expert_banks):
            raise IndexError(f"state expert bank is unavailable for domain {domain_id}")
        if len(self.state_expert_banks[domain_id]) != len(
            self.regime_router.state_prototypes[domain_id]
        ):
            raise RuntimeError("state expert bank and router prototype memory are out of sync")
        routing_descriptors = descriptors.detach().float().cpu()
        decisions: list[RoutingDecision] = []
        for descriptor in routing_descriptors:
            decision = self.regime_router.route_state(
                descriptor,
                domain_id,
                allow_create=allow_create,
            )
            if decision.action == CREATE_STATE:
                created_id = self._create_state_expert(domain_id)
                if created_id != decision.selected_id:
                    raise RuntimeError("router and state bank creation ids diverged")
            self.regime_router.commit_state(
                decision,
                descriptor,
                domain_id,
                update_prototype=update_prototype,
                record_usage=record_usage,
            )
            decisions.append(decision)
        return decisions

    def set_residual_normalization(self, residual_mean: float, residual_std: float) -> None:
        std = max(float(residual_std), 1e-6)
        zero = -float(residual_mean) / std
        self.residual_zero_normalized = torch.full(
            (1, self.horizon, 1),
            zero,
            device=self.residual_zero_normalized.device,
            dtype=self.residual_zero_normalized.dtype,
        )

    def horizon_gate_values(self) -> list[float]:
        return [float(value) for value in torch.sigmoid(self.horizon_gate_logits.detach()).cpu().tolist()]

    def domain_gate_values(self) -> list[float]:
        return [
            float(torch.sigmoid(gate.detach()).cpu())
            for gate in self.domain_experts.gate_logits
        ]

    def state_gate_values(self) -> list[list[float]]:
        return [
            [float(torch.sigmoid(gate.detach()).cpu()) for gate in bank.gate_logits]
            for bank in self.state_expert_banks
        ]

    def get_extra_state(self) -> dict[str, Any]:
        state = super().get_extra_state()
        if hasattr(self, "flow_embedding_input_length"):
            state["canonical_io"] = self._canonical_io_state()
        state["hierarchical_routing"] = {
            "router": self.regime_router.export_state(),
            "num_domain_experts": len(self.domain_experts),
            "num_state_experts": [len(bank) for bank in self.state_expert_banks],
            "adapter_bottleneck": self.adapter_bottleneck,
            "adapter_dropout": self.adapter_dropout,
            "domain_descriptor_max_windows": self.domain_descriptor_max_windows,
        }
        return state

    def _canonical_io_state(self) -> dict[str, Any]:
        return {
            "base_input_length": self.flow_embedding_input_length,
            "extra_input_lengths": [
                int(key[1:])
                for key in self.flow_embedding_bank.keys()
            ],
            "base_horizon": int(self.base_horizon),
            "extra_horizons": [
                int(key[1:])
                for key in self.canonical_head_bank.keys()
            ],
            "active_input_length": self.active_input_length,
            "active_horizon": int(self.active_horizon),
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        super().set_extra_state(dict(state))
        io_state = state.get("canonical_io")

        if io_state is not None:
            device = self.series_embedding.device
            dtype = self.series_embedding.dtype

            base_input_length = io_state.get(
                "base_input_length"
            )

            if base_input_length is not None:
                base_input_length = int(
                    base_input_length
                )

                if (
                        self.flow_embedding is None
                        or int(self.flow_embedding.in_features)
                        != base_input_length
                ):
                    self.flow_embedding = nn.Linear(
                        base_input_length,
                        self.d_model,
                    ).to(
                        device=device,
                        dtype=dtype,
                    )

                self.flow_embedding_input_length = (
                    base_input_length
                )

            self.flow_embedding_bank = nn.ModuleDict()

            for input_length in io_state.get(
                    "extra_input_lengths",
                    [],
            ):
                input_length = int(input_length)
                key = f"T{input_length}"

                self.flow_embedding_bank[key] = nn.Linear(
                    input_length,
                    self.d_model,
                ).to(
                    device=device,
                    dtype=dtype,
                )

            saved_base_horizon = int(
                io_state.get(
                    "base_horizon",
                    self.base_horizon,
                )
            )

            if saved_base_horizon != self.base_horizon:
                raise ValueError(
                    "checkpoint base horizon does not match "
                    f"model config: checkpoint="
                    f"{saved_base_horizon}, "
                    f"model={self.base_horizon}"
                )

            self.canonical_head_bank = nn.ModuleDict()

            for horizon in io_state.get(
                    "extra_horizons",
                    [],
            ):
                horizon = int(horizon)
                key = f"H{horizon}"

                self.canonical_head_bank[key] = nn.Linear(
                    self.d_model,
                    horizon,
                ).to(
                    device=device,
                    dtype=dtype,
                )

            active_input_length = io_state.get(
                "active_input_length"
            )

            self.active_input_length = (
                None
                if active_input_length is None
                else int(active_input_length)
            )

            self.active_horizon = int(
                io_state.get(
                    "active_horizon",
                    self.base_horizon,
                )
            )
        hierarchy = state.get("hierarchical_routing")
        if hierarchy is None:
            return
        if int(hierarchy.get("adapter_bottleneck", self.adapter_bottleneck)) != self.adapter_bottleneck:
            raise ValueError("checkpoint adapter_bottleneck does not match the model config")
        saved_dropout = float(hierarchy.get("adapter_dropout", self.adapter_dropout))
        if abs(saved_dropout - self.adapter_dropout) > 1e-12:
            raise ValueError("checkpoint adapter dropout does not match the model config")
        self.domain_descriptor_max_windows = int(
            hierarchy.get("domain_descriptor_max_windows", self.domain_descriptor_max_windows)
        )
        device = self.series_embedding.device
        dtype = self.series_embedding.dtype
        domain_count = int(hierarchy["num_domain_experts"])
        state_counts = [int(value) for value in hierarchy["num_state_experts"]]
        if len(state_counts) != domain_count:
            raise ValueError("checkpoint state-bank count does not match domain count")
        self.domain_experts.rebuild(domain_count, device=device, dtype=dtype)
        self.state_expert_banks = nn.ModuleList()
        for state_count in state_counts:
            bank = self._new_state_bank().to(device=device, dtype=dtype)
            bank.rebuild(state_count, device=device, dtype=dtype)
            self.state_expert_banks.append(bank)
        self.regime_router.load_exported_state(hierarchy["router"])
        if len(self.regime_router.domain_prototypes) != domain_count:
            raise ValueError("checkpoint domain prototype count does not match expert modules")
        if [len(values) for values in self.regime_router.state_prototypes] != state_counts:
            raise ValueError("checkpoint state prototype counts do not match expert modules")

    def _apply_state_experts(
            self,
            domain_hidden: torch.Tensor,
            domain_id: int,
    ) -> tuple[torch.Tensor, list[int]]:
        if domain_id >= len(self.state_expert_banks):
            raise RuntimeError("selected domain has no TrafficStateExpertBank")

        bank = self.state_expert_banks[domain_id]

        # ---------------------------------------------------------
        # Replay path:
        # use the historical State Expert assignment saved in memory.
        # Do NOT reroute through current state prototypes.
        # ---------------------------------------------------------
        if self._forced_state_ids is not None:
            state_ids = [int(value) for value in self._forced_state_ids]

            if len(state_ids) != int(domain_hidden.shape[0]):
                raise RuntimeError(
                    "forced replay state ids do not align with batch size: "
                    f"state_ids={len(state_ids)} batch={domain_hidden.shape[0]}"
                )

            for state_id in state_ids:
                if state_id < 0 or state_id >= len(bank):
                    raise IndexError(
                        "forced replay state expert is unavailable: "
                        f"domain={domain_id} state={state_id} "
                        f"num_states={len(bank)}"
                    )

            adapted_by_row = [
                bank(
                    domain_hidden[position: position + 1],
                    state_id,
                )
                for position, state_id in enumerate(state_ids)
            ]

            output = (
                torch.cat(adapted_by_row, dim=0)
                if adapted_by_row
                else domain_hidden
            )

            return output, state_ids

        # ---------------------------------------------------------
        # Normal training / inference path:
        # route according to the current Domain -> State router.
        # ---------------------------------------------------------
        state_descriptors = self.regime_router.pool_windows(domain_hidden)

        allow_create = bool(
            self.training and self.routing_updates_enabled
        )
        update_prototype = bool(
            self.training and self.routing_updates_enabled
        )

        decisions = self.route_state_descriptors(
            state_descriptors,
            domain_id,
            allow_create=allow_create,
            update_prototype=update_prototype,
            record_usage=update_prototype,
        )

        state_ids = [
            decision.selected_id
            for decision in decisions
        ]

        adapted_by_row = [
            bank(
                domain_hidden[position: position + 1],
                state_id,
            )
            for position, state_id in enumerate(state_ids)
        ]

        output = (
            torch.cat(adapted_by_row, dim=0)
            if adapted_by_row
            else domain_hidden
        )

        return output, state_ids

    def _create_domain_expert(self) -> int:
        device = self.series_embedding.device
        dtype = self.series_embedding.dtype
        domain_id = self.domain_experts.create_expert(device=device, dtype=dtype)
        self.state_expert_banks.append(self._new_state_bank().to(device=device, dtype=dtype))
        return domain_id

    def _create_state_expert(self, domain_id: int) -> int:
        device = self.series_embedding.device
        dtype = self.series_embedding.dtype
        return self.state_expert_banks[int(domain_id)].create_expert(
            device=device,
            dtype=dtype,
        )

    def _new_state_bank(self) -> TrafficStateExpertBank:
        return TrafficStateExpertBank(
            d_model=self.d_model,
            bottleneck=self.adapter_bottleneck,
            dropout=self.adapter_dropout,
        )



class CanonicalCrossNodeBlock(nn.Module):
    """Cross-node block copied from the successful router diagnostic."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.norm1(x)
        # Forecasting does not consume the quadratic attention matrix. Opt in
        # only for diagnostics, allowing fused attention on large node sets.
        attended, attention = self.attn(q, q, q, need_weights=getattr(self, "capture_attention", False), average_attn_weights=False)
        x = x + self.drop1(attended)
        x = x + self.ffn(self.norm2(x))
        return x, attention


class ExpertLiteInvertedTrafficTransformer(ExpertLightweightTrafficTransformer):
    """Canonical TraffiCL model migrated from the successful RoutedBackbone diagnostic."""

    def __init__(
        self,
        *args,
        enable_cross_node: bool = True,
        use_dynamic_node_ids: bool = False,
        enable_state_expert: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.horizon_gate_logits.requires_grad_(False)
        for parameter in self.sample_gate.parameters():
            parameter.requires_grad_(False)
        self.enable_cross_node = bool(enable_cross_node)
        self.use_dynamic_node_ids = False
        self.enable_state_expert = bool(enable_state_expert)
        self.input_window_eps = 1e-5
        self.flow_embedding: nn.Linear | None = None
        self.flow_embedding_input_length: int | None = None
        self.flow_embedding_bank = nn.ModuleDict()

        self.cross_blocks = nn.ModuleList(
            [
                CanonicalCrossNodeBlock(
                    d_model=self.d_model,
                    n_heads=4 if self.d_model % 4 == 0 else 1,
                    d_ff=max(self.d_model * 2, self.d_model),
                    dropout=self.adapter_dropout,
                )
                for _ in range(2)
            ]
        )

        self.canonical_norm = nn.LayerNorm(self.d_model)

        # Keep the original head as the base horizon head so that the
        # original single-dataset protocol remains compatible.
        self.canonical_head = nn.Linear(self.d_model, self.horizon)
        self.base_horizon = int(self.horizon)

        # Additional horizon-specific heads, e.g. H1 / H4.
        self.canonical_head_bank = nn.ModuleDict()

        self.active_input_length: int | None = None
        self.active_horizon: int = int(self.horizon)

        self.last_representation_stats: dict[str, float] = {}
        self.last_attention_stats: dict[str, float] = {}
        self.last_attention_matrix: torch.Tensor | None = None
        self.last_window_stats: dict[str, float] = {}

    def routing_summary(self) -> dict[str, Any]:
        summary = super().routing_summary()
        summary.pop("horizon_gates", None)
        return summary

    def _ensure_flow_embedding(
            self,
            input_length: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> nn.Linear:
        input_length = int(input_length)

        if input_length <= 0:
            raise ValueError(
                f"input_length must be positive, got {input_length}"
            )

        # First temporal shape: keep the original parameter name.
        if self.flow_embedding is None:
            self.flow_embedding = nn.Linear(
                input_length,
                self.d_model,
            ).to(
                device=device,
                dtype=dtype,
            )
            self.flow_embedding_input_length = input_length
            return self.flow_embedding

        if self.flow_embedding_input_length is None:
            self.flow_embedding_input_length = int(
                self.flow_embedding.in_features
            )

        # Reuse the original embedding for the base input length.
        if input_length == self.flow_embedding_input_length:
            return self.flow_embedding

        # Different input length: do NOT replace the old embedding.
        # Create/reuse a persistent lightweight temporal adapter.
        key = f"T{input_length}"

        if key not in self.flow_embedding_bank:
            adapter = nn.Linear(
                input_length,
                self.d_model,
            ).to(
                device=device,
                dtype=dtype,
            )

            assert self.flow_embedding is not None

            with torch.no_grad():
                base_weight = self.flow_embedding.weight.detach().to(
                    device=device,
                    dtype=dtype,
                )

                # [D, T_base] -> [D, 1, T_base]
                weight_3d = base_weight.unsqueeze(1)

                resized_weight = F.interpolate(
                    weight_3d,
                    size=input_length,
                    mode="linear",
                    align_corners=True,
                ).squeeze(1)

                # Keep approximate projection magnitude comparable when
                # the temporal length changes.
                base_length = int(
                    self.flow_embedding.in_features
                )

                resized_weight = resized_weight * (
                        float(base_length)
                        / float(input_length)
                )

                adapter.weight.copy_(resized_weight)

                if (
                        adapter.bias is not None
                        and self.flow_embedding.bias is not None
                ):
                    adapter.bias.copy_(
                        self.flow_embedding.bias.detach().to(
                            device=device,
                            dtype=dtype,
                        )
                    )

            self.flow_embedding_bank[key] = adapter

            print(
                f"[Global IO] CREATE aligned input adapter "
                f"{key}: {input_length} -> {self.d_model} "
                f"from base T{self.flow_embedding.in_features}",
                flush=True,
            )

        return self.flow_embedding_bank[key]

    def _ensure_forecast_head(
            self,
            horizon: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> nn.Linear:
        horizon = int(horizon)

        if horizon <= 0:
            raise ValueError(
                f"horizon must be positive, got {horizon}"
            )

        # Base forecasting horizon keeps the original head.
        if horizon == self.base_horizon:
            return self.canonical_head

        key = f"H{horizon}"

        if key not in self.canonical_head_bank:
            self.canonical_head_bank[key] = nn.Linear(
                self.d_model,
                horizon,
            ).to(
                device=device,
                dtype=dtype,
            )

            print(
                f"[Global IO] CREATE forecast head "
                f"{key}: {self.d_model} -> {horizon}",
                flush=True,
            )

        return self.canonical_head_bank[key]

    def configure_io_shape(
            self,
            input_length: int,
            horizon: int,
    ) -> None:
        """
        Select the dataset-specific temporal I/O adapters while
        preserving the shared Transformer and hierarchical experts.
        """
        input_length = int(input_length)
        horizon = int(horizon)

        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype

        self._ensure_flow_embedding(
            input_length,
            device=device,
            dtype=dtype,
        )

        self._ensure_forecast_head(
            horizon,
            device=device,
            dtype=dtype,
        )

        self.active_input_length = input_length
        self.active_horizon = horizon

    def _input_window_norm(self, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = flow.mean(dim=1, keepdim=True).detach()
        centered = flow - mean
        std = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + self.input_window_eps)
        return centered / std, mean, std

    def _flow_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected x [B,T,N,C], got {tuple(x.shape)}")
        batch_size, input_length, num_series, input_dim = x.shape
        if input_dim < 1:
            raise ValueError(
                f"TraffiCL requires at least one flow feature, got input_dim={input_dim}"
            )

        self.active_input_length = int(input_length)
        if int(self._active_node_indices.numel()) != num_series:
            self.expand_node_params(num_series)
        flow_embedding = self._ensure_flow_embedding(
            input_length,
            device=x.device,
            dtype=x.dtype,
        )

        flow = x[..., 0]
        flow_norm, mean, std = self._input_window_norm(flow)

        tokens = flow_embedding(
            flow_norm.permute(0, 2, 1)
        )
        self.last_window_stats = {
            "flow_mean": float(flow.detach().mean().cpu()),
            "flow_std": float(flow.detach().std(unbiased=False).cpu()),
            "window_mean_mean": float(mean.detach().mean().cpu()),
            "window_std_mean": float(std.detach().mean().cpu()),
        }
        return tokens, mean, std, flow_norm

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _, _, _ = self._flow_tokens(x)
        hidden = tokens
        attentions = []
        if self.enable_cross_node:
            for block in self.cross_blocks:
                hidden, attention = block(hidden)
                if attention is not None:
                    attentions.append(attention.detach())
        hidden = self.canonical_norm(hidden)
        self.last_representation_stats = {
            "token_norm": float(tokens.detach().norm().cpu()),
            "hidden_norm": float(hidden.detach().norm().cpu()),
            "hidden_mean": float(hidden.detach().mean().cpu()),
            "hidden_std": float(hidden.detach().std(unbiased=False).cpu()),
        }
        if attentions:
            attention = torch.stack(attentions, dim=0).mean(dim=0)
            self.last_attention_matrix = attention.detach().cpu()
            self.last_attention_stats = LiteInvertedTrafficTransformer._attention_stats(attention)
        else:
            self.last_attention_matrix = None
            self.last_attention_stats = {}
        return hidden

    def forecast_from_hidden(
            self,
            hidden: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(
                f"expected hidden [B,N,D], got {tuple(hidden.shape)}"
            )

        horizon = int(self.active_horizon)

        head = self._ensure_forecast_head(
            horizon,
            device=hidden.device,
            dtype=hidden.dtype,
        )

        return head(hidden).permute(
            0,
            2,
            1,
        ).contiguous()

    def forward_with_components(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[int] | int]]:
        domain_id = self.active_domain_id
        if domain_id is None:
            raise RuntimeError("no active traffic domain; compute one task-level descriptor and route it first")
        if domain_id >= len(self.domain_experts):
            raise RuntimeError("active domain is missing from TrafficDomainExpertBank")
        tokens, mean, std, flow_norm = self._flow_tokens(x)
        hidden = tokens
        attentions = []
        if self.enable_cross_node:
            for block in self.cross_blocks:
                hidden, attention = block(hidden)
                if attention is not None:
                    attentions.append(attention.detach())
        backbone_hidden = self.canonical_norm(hidden)
        self.last_representation_stats = {
            "token_norm": float(tokens.detach().norm().cpu()),
            "hidden_norm": float(backbone_hidden.detach().norm().cpu()),
            "hidden_mean": float(backbone_hidden.detach().mean().cpu()),
            "hidden_std": float(backbone_hidden.detach().std(unbiased=False).cpu()),
        }
        if attentions:
            attention = torch.stack(attentions, dim=0).mean(dim=0)
            self.last_attention_matrix = attention.detach().cpu()
            self.last_attention_stats = LiteInvertedTrafficTransformer._attention_stats(attention)
        else:
            self.last_attention_matrix = None
            self.last_attention_stats = {}
        backbone_forecast_norm = self.forecast_from_hidden(backbone_hidden)
        backbone_forecast = backbone_forecast_norm * std + mean
        domain_hidden = self.domain_experts(backbone_hidden, domain_id)
        if self.enable_state_expert:
            state_hidden, state_ids = self._apply_state_experts(domain_hidden, domain_id)
        else:
            state_hidden, state_ids = domain_hidden, []
        final_forecast_norm = self.forecast_from_hidden(state_hidden)
        final_forecast = final_forecast_norm * std + mean
        domain_gate = self.domain_experts.gate_value(domain_id).to(device=x.device, dtype=x.dtype)
        state_gate_values = [
            self.state_expert_banks[domain_id].gate_value(state_id).to(device=x.device, dtype=x.dtype)
            for state_id in state_ids
        ]
        state_gate = torch.stack(state_gate_values).mean() if state_gate_values else torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
        self.last_routing_trace = {
            "backbone_hidden_shape": tuple(backbone_hidden.shape),
            "domain_hidden_shape": tuple(domain_hidden.shape),
            "state_hidden_shape": tuple(state_hidden.shape),
            "state_descriptor_source": "domain_hidden",
            "domain_id": domain_id,
            "state_ids": list(state_ids),
            "domain_gate": float(domain_gate.detach().cpu()),
            "state_gate": float(state_gate.detach().cpu()),
            "flow_only": True,
            "input_window_norm": True,
            "dynamic_node_ids": False,
        }
        return final_forecast, {
            "input_flow_norm": flow_norm,
            "window_mean": mean,
            "window_std": std,
            "token_embedding": tokens,
            "backbone_hidden": backbone_hidden,
            "backbone_forecast": backbone_forecast,
            "domain_hidden": domain_hidden,
            "state_hidden": state_hidden,
            "state_ids": state_ids,
            "domain_id": domain_id,
            "expert_contribution": final_forecast - backbone_forecast,
            "domain_gate": domain_gate,
            "state_gate": state_gate,
        }

    def delta_stats(self) -> dict[str, float]:
        if not self.last_routing_trace:
            return {"domain_delta_ratio": float("nan"), "state_delta_ratio": float("nan")}
        # The successful diagnostic reports representation deltas. Recompute from
        # the latest forward components stored through autograd-safe detached refs.
        return {
            "domain_delta_ratio": float("nan"),
            "state_delta_ratio": float("nan"),
        }

