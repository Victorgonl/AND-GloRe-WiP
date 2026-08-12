from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass
class TrainingArgs:
    seeds: tuple[int, ...]
    epochs: int
    lr: float
    l2_coef: float
    feat_mask: float
    adj_mask: float
    hidden_dim: int
    embed_dim: int
    gnn_hidden_dim: int
    proj_dim: int
    temperature: float
    dropout: float
    gnn_dropout: float
    gnn_alpha: float
    min_distance_threshold: float
    max_distance_threshold: float
    step: Optional[float]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __str__(self) -> str:
        return "\n".join(
            f"{field.name}: {getattr(self, field.name)}" for field in fields(self)
        )
