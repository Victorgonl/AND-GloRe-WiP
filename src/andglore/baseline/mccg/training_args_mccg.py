from dataclasses import dataclass, fields

import torch


@dataclass
class TrainingArgsMCCG:
    seeds: tuple[int, ...]
    epochs: int
    lr: float
    layer_shape: tuple[int, int, int]
    dim_proj_multiview: int
    dim_proj_cluster: int
    drop_scheme: str
    drop_feature_rate_view1: float
    drop_feature_rate_view2: float
    drop_edge_rate_view1: float
    drop_edge_rate_view2: float
    th_a: int
    th_o: float
    th_v: int
    db_eps: float
    db_min: int
    l2_coef: float
    w_cluster: float
    t_multiview: float
    t_cluster: float
    gat_alpha: float = 0.2
    gat_dropout: float = 0.6
    diffusion_steps: int = 2
    augmentation_threshold: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        if len(self.layer_shape) != 3:
            raise ValueError("layer_shape must contain [input, hidden, output] dimensions")
        if self.drop_scheme not in {"degree", "pr", "evc"}:
            raise ValueError("drop_scheme must be one of: degree, pr, evc")
        if self.db_min < 2:
            raise ValueError("db_min must be at least 2 for HDBSCAN")
        if not 0 <= self.w_cluster <= 1:
            raise ValueError("w_cluster must be between 0 and 1")

    def __str__(self) -> str:
        return "\n".join(
            f"{field.name}: {getattr(self, field.name)}" for field in fields(self)
        )
