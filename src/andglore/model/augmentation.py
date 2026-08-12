from typing import Sequence

import torch


def get_augmented_features(
    feature_matrix: torch.Tensor,
    feature_drop_rate: float,
) -> torch.Tensor:
    """
    Randomly mask features in the feature matrix.
    """

    features = feature_matrix.clone()
    if feature_drop_rate > 0.0:
        keep_mask = torch.rand_like(features) >= feature_drop_rate
        features *= keep_mask

    return features


def get_augmented_adjacencies(
    adjacencies: Sequence[torch.Tensor],
    edge_drop_rate: float,
    hub_strength: float = 1.0,
) -> list[torch.Tensor]:

    def _symmetrize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        """Return a symmetrized copy of an adjacency matrix."""

        return (adj + adj.t()) / 2

    symmetric_adjacencies = [
        _symmetrize_adjacency(adjacency) for adjacency in adjacencies
    ]

    # The mean graph is the shared structural reference used to identify hubs
    # and sample a single edge-retention mask for all metapaths.
    reference_adjacency = torch.stack(symmetric_adjacencies, dim=0).mean(dim=0)

    degree = reference_adjacency.sum(dim=1)
    normalized_degree = degree / degree.max().clamp_min(1e-12)
    edge_score = (normalized_degree.unsqueeze(0) + normalized_degree.unsqueeze(1)) / 2
    drop_probability = (edge_drop_rate * (1 + hub_strength * edge_score)).clamp(0, 0.95)

    shared_keep_mask = torch.rand_like(reference_adjacency) >= drop_probability
    shared_keep_mask = torch.triu(shared_keep_mask, diagonal=1)
    shared_keep_mask = shared_keep_mask | shared_keep_mask.t()

    augmented_adjacencies = []
    for adjacency in symmetric_adjacencies:
        masked_adjacency = adjacency * shared_keep_mask
        augmented_adjacencies.append(masked_adjacency)

    return augmented_adjacencies
