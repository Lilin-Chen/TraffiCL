from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class LightweightTrafficTransformer(nn.Module):
    """Lightweight traffic forecaster with temporal and station mixing.

    Input shape:  [batch, input_length, num_series, input_dim]
    Output shape: [batch, horizon, num_series]
    """

    def __init__(
        self,
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
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.input_dim = input_dim
        self.horizon = horizon
        self.max_series = max_series or num_series
        self.use_spatial_mixer = use_spatial_mixer
        if self.max_series < num_series:
            raise ValueError("max_series must be greater than or equal to num_series")

        initial_ids = (
            [str(value) for value in initial_node_ids]
            if initial_node_ids is not None
            else [f"__anonymous_node_{index}" for index in range(num_series)]
        )
        if len(initial_ids) != num_series:
            raise ValueError("initial_node_ids length must match num_series")
        if len(set(initial_ids)) != len(initial_ids):
            raise ValueError("initial_node_ids must be unique")
        self._node_ids = list(initial_ids)
        self._node_index = {node_id: index for index, node_id in enumerate(initial_ids)}
        self.num_series = len(initial_ids)
        self.register_buffer(
            "_active_node_indices",
            torch.arange(self.num_series, dtype=torch.long),
            persistent=False,
        )

        self.input_projection = nn.Linear(input_dim, d_model)
        self.series_embedding = nn.Parameter(torch.empty(1, 1, self.max_series, d_model))
        nn.init.normal_(self.series_embedding, mean=0.0, std=0.02)

        self.position = SinusoidalPositionalEncoding(d_model)
        self.local_mixer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.temporal_norm = nn.LayerNorm(d_model)

        if use_spatial_mixer:
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.spatial_dropout = nn.Dropout(dropout)
            self.spatial_norm = nn.LayerNorm(d_model)
            self.spatial_ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            self.spatial_ffn_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forecast_from_hidden(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input window into shared traffic representations [B, N, D]."""

        if x.ndim != 4:
            raise ValueError(f"expected x with 4 dims [B,T,N,C], got shape {tuple(x.shape)}")

        batch_size, input_length, num_series, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {input_dim}")
        if num_series > self.max_series:
            raise ValueError(f"received {num_series} series, but max_series={self.max_series}")

        if int(self._active_node_indices.numel()) != num_series:
            self.expand_node_params(num_series)
        active_indices = self._active_node_indices.to(self.series_embedding.device)
        station_embedding = self.series_embedding.index_select(2, active_indices)

        hidden = self._project_inputs(x)
        hidden = hidden + station_embedding
        hidden = hidden.permute(0, 2, 1, 3).reshape(batch_size * num_series, input_length, -1)
        hidden = self.position(hidden)

        local_hidden = self.local_mixer(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden + local_hidden
        hidden = self.encoder(hidden)
        hidden = self.temporal_norm(hidden[:, -1])
        hidden = hidden.reshape(batch_size, num_series, -1)

        if self.use_spatial_mixer:
            spatial_hidden, _ = self.spatial_attention(hidden, hidden, hidden, need_weights=False)
            hidden = self.spatial_norm(hidden + self.spatial_dropout(spatial_hidden))
            hidden = self.spatial_ffn_norm(hidden + self.spatial_ffn(hidden))

        return hidden

    def forecast_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the existing forecast head to shared or adapted [B, N, D] features."""

        if hidden.ndim != 3:
            raise ValueError(
                f"expected hidden with 3 dims [B,N,D], got shape {tuple(hidden.shape)}"
            )

        forecast = self.head(hidden)
        return forecast.permute(0, 2, 1).contiguous()

    def _project_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_projection(x)

    def configure_node_ids(self, node_ids: Sequence[str]) -> None:
        """Select station embeddings by stable identity across continual tasks."""

        normalized = [str(value) for value in node_ids]
        if not normalized:
            raise ValueError("node_ids must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("node_ids must be unique within a task")

        unseen = [node_id for node_id in normalized if node_id not in self._node_index]
        if len(self._node_ids) + len(unseen) > self.max_series:
            raise ValueError(
                "stable station vocabulary exceeds max_series: "
                f"required={len(self._node_ids) + len(unseen)} max_series={self.max_series}"
            )
        for node_id in unseen:
            self._node_index[node_id] = len(self._node_ids)
            self._node_ids.append(node_id)

        self._active_node_indices = torch.tensor(
            [self._node_index[node_id] for node_id in normalized],
            dtype=torch.long,
            device=self._active_node_indices.device,
        )
        self.num_series = len(normalized)

    def expand_node_params(self, new_num_series: int) -> None:
        """Backward-compatible positional activation for datasets without IDs."""

        new_num_series = int(new_num_series)
        if new_num_series <= 0 or new_num_series > self.max_series:
            raise ValueError(
                f"new_num_series must be in [1, {self.max_series}], got {new_num_series}"
            )
        while len(self._node_ids) < new_num_series:
            node_id = f"__anonymous_node_{len(self._node_ids)}"
            self._node_index[node_id] = len(self._node_ids)
            self._node_ids.append(node_id)
        self._active_node_indices = torch.arange(
            new_num_series,
            dtype=torch.long,
            device=self._active_node_indices.device,
        )
        self.num_series = new_num_series

    def get_extra_state(self) -> dict[str, Any]:
        active_ids = [self._node_ids[index] for index in self._active_node_indices.tolist()]
        return {"node_ids": list(self._node_ids), "active_node_ids": active_ids}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        node_ids = [str(value) for value in state.get("node_ids", [])]
        if not node_ids:
            return
        if len(node_ids) > self.max_series:
            raise ValueError("checkpoint station vocabulary exceeds max_series")
        self._node_ids = node_ids
        self._node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        active_ids = [str(value) for value in state.get("active_node_ids", node_ids)]
        self._active_node_indices = torch.tensor(
            [self._node_index[node_id] for node_id in active_ids],
            dtype=torch.long,
            device=self._active_node_indices.device,
        )
        self.num_series = len(active_ids)



class LiteInvertedTrafficTransformer(LightweightTrafficTransformer):
    """Node-token traffic forecaster for structure validation.

    Input:  [B, T, N, C]
    Output: [B, H, N]
    Hidden: [B, N, D]
    """

    def __init__(
        self,
        *args,
        enable_cross_node: bool = True,
        use_dynamic_node_ids: bool = True,
        ffn_ratio: int = 2,
        **kwargs,
    ):
        super().__init__(*args, use_spatial_mixer=False, **kwargs)
        self.enable_cross_node = bool(enable_cross_node)
        self.use_dynamic_node_ids = bool(use_dynamic_node_ids)
        self.temporal_depthwise = nn.Conv1d(
            self.input_dim,
            self.input_dim,
            kernel_size=3,
            padding=1,
            groups=self.input_dim,
        )
        self.temporal_pointwise = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.GELU(),
            nn.Linear(self.input_dim, self.input_projection.out_features),
        )
        d_model = self.input_projection.out_features
        dropout = self.head[2].p if isinstance(self.head, nn.Sequential) else 0.1
        self.cross_blocks = nn.ModuleList(
            [
                LiteCrossNodeBlock(
                    d_model=d_model,
                    nhead=4 if d_model % 4 == 0 else 1,
                    dim_feedforward=max(d_model * int(ffn_ratio), d_model),
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        )
        self.fusion_gamma = nn.Parameter(torch.tensor(1.0))
        self.fusion_norm = nn.LayerNorm(d_model)
        self.last_representation_stats: dict[str, float] = {}
        self.last_attention_stats: dict[str, float] = {}
        self.last_attention_matrix: torch.Tensor | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected x [B,T,N,C], got {tuple(x.shape)}")
        batch_size, input_length, num_series, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {input_dim}")
        if num_series > self.max_series:
            raise ValueError(f"received {num_series} series, but max_series={self.max_series}")
        if int(self._active_node_indices.numel()) != num_series:
            self.expand_node_params(num_series)

        node_series = x.permute(0, 2, 3, 1).reshape(batch_size * num_series, input_dim, input_length)
        local = self.temporal_depthwise(node_series)
        local = local.mean(dim=-1).reshape(batch_size, num_series, input_dim)
        z_temp = self.temporal_pointwise(local)

        active_indices = self._active_node_indices.to(self.series_embedding.device)
        if self.use_dynamic_node_ids:
            z_token = z_temp + self.series_embedding.index_select(2, active_indices).squeeze(1)
        else:
            z_token = z_temp

        if self.enable_cross_node:
            z_cross = z_token
            attentions = []
            for block in self.cross_blocks:
                z_cross, attention = block(z_cross)
                attentions.append(attention.detach())
        else:
            z_cross = torch.zeros_like(z_temp)
            attentions = []

        hidden = self.fusion_norm(z_temp + self.fusion_gamma.to(z_temp.dtype) * z_cross)
        self.last_representation_stats = self._representation_stats(z_temp, z_cross, hidden)
        if attentions:
            attention = torch.stack(attentions, dim=0).mean(dim=0)
            self.last_attention_matrix = attention.detach().cpu()
            self.last_attention_stats = self._attention_stats(attention)
        else:
            self.last_attention_matrix = None
            self.last_attention_stats = {}
        return hidden

    def _representation_stats(self, z_temp: torch.Tensor, z_cross: torch.Tensor, hidden: torch.Tensor) -> dict[str, float]:
        z_temp_norm = float(z_temp.detach().norm(dim=-1).mean().cpu())
        z_cross_norm = float(z_cross.detach().norm(dim=-1).mean().cpu())
        return {
            "z_temp_norm": z_temp_norm,
            "z_temp_std": float(z_temp.detach().std().cpu()),
            "z_cross_norm": z_cross_norm,
            "cross_to_temporal_norm_ratio": z_cross_norm / max(z_temp_norm, 1e-12),
            "fusion_gamma": float(self.fusion_gamma.detach().cpu()),
            "H_norm": float(hidden.detach().norm(dim=-1).mean().cpu()),
        }

    @staticmethod
    def _attention_stats(attention: torch.Tensor) -> dict[str, float]:
        # attention shape: [B, heads, N, N]
        probs = attention.detach().float().clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1).mean()
        n = probs.shape[-1]
        eye = torch.eye(n, device=probs.device, dtype=torch.bool).view(1, 1, n, n)
        diagonal = probs.masked_select(eye).mean() if n else probs.new_tensor(float("nan"))
        off_diagonal = probs.masked_select(~eye).mean() if n > 1 else probs.new_tensor(float("nan"))
        topk = torch.topk(probs, k=min(5, n), dim=-1).values
        return {
            "attention_entropy": float(entropy.cpu()),
            "diagonal_attention_mass": float(diagonal.cpu()),
            "off_diagonal_attention_mass": float(off_diagonal.cpu()),
            "top1_attention_mass": float(topk[..., :1].sum(dim=-1).mean().cpu()),
            "top5_attention_mass": float(topk.sum(dim=-1).mean().cpu()),
            "attention_max": float(probs.max().cpu()),
            "attention_mean": float(probs.mean().cpu()),
        }


class LiteCrossNodeBlock(nn.Module):
    def __init__(self, *, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.norm1(x)
        attn_out, weights = self.attn(q, q, q, need_weights=True, average_attn_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x, weights
