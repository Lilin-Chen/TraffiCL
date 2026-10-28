from __future__ import annotations
from typing import Any, Mapping
from data.schema import WindowBatch
from models.backbones.expert_lightweight_transformer import ExpertLiteInvertedTrafficTransformer

def build_model(config: Mapping[str, Any], sample_batch: WindowBatch):
    model_config = dict(config.get("model", {}))
    model_name = str(model_config.get("name", "expert_lite_inverted_traffic")).lower()
    if model_name == "expert_lite_inverted_traffic":
        return ExpertLiteInvertedTrafficTransformer(
            input_dim=sample_batch.x.shape[-1],
            horizon=sample_batch.y.shape[1],
            num_series=sample_batch.x.shape[2],
            d_model=int(model_config.get("d_model", 32)),
            nhead=int(model_config.get("nhead", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
            dim_feedforward=int(model_config.get("dim_feedforward", 128)),
            dropout=float(model_config.get("dropout", 0.1)),
            max_series=int(model_config.get("max_series", sample_batch.x.shape[2])),
            use_spatial_mixer=False,
            initial_node_ids=sample_batch.series_ids,
            adapter_bottleneck=int(model_config.get("adapter_bottleneck", 16)),
            feature_groups=model_config.get("feature_groups"),
            dynamic_domain_experts=bool(model_config.get("dynamic_domain_experts", True)),
            dynamic_state_experts=bool(model_config.get("dynamic_state_experts", True)),
            max_domain_experts=int(model_config.get("max_domain_experts", 8)),
            max_state_experts_per_domain=int(model_config.get("max_state_experts_per_domain", 8)),
            domain_similarity_threshold=float(model_config.get("domain_similarity_threshold", 0.85)),
            state_similarity_threshold=float(model_config.get("state_similarity_threshold", 0.85)),
            domain_prototype_update_rate=float(model_config.get("domain_prototype_update_rate", 0.1)),
            state_prototype_update_rate=float(model_config.get("state_prototype_update_rate", 0.1)),
            domain_descriptor_max_windows=int(model_config.get("domain_descriptor_max_windows", 2048)),
            enable_cross_node=bool(model_config.get("enable_cross_node", True)),
            use_dynamic_node_ids=bool(model_config.get("use_dynamic_node_ids", True)),
            enable_state_expert=bool(model_config.get("enable_state_expert", True)),
        )
    raise ValueError(f"This release supports only the verified TraffiCL V5 model: {model_name}")
