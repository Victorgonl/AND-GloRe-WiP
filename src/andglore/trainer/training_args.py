from dataclasses import dataclass, fields

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
    min_distance_threshold: float = 0.1
    max_distance_threshold: float = 0.3
    step: float | None = 0.05
    clustering_algorithm: str = "hac"
    db_eps: float = 0.0
    db_min: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.clustering_algorithm = self.clustering_algorithm.lower()
        if self.clustering_algorithm not in {"hac", "hdbscan"}:
            raise ValueError("clustering_algorithm must be one of: hac, hdbscan")
        if self.db_eps < 0:
            raise ValueError("db_eps must be non-negative")
        if self.db_min < 2:
            raise ValueError("db_min must be at least 2 for HDBSCAN")

    def __str__(self) -> str:
        return "\n".join(
            f"{field.name}: {getattr(self, field.name)}" for field in fields(self)
        )
