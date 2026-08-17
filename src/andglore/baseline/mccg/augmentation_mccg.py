import networkx as nx
import torch
import torch.nn.functional as F


def adjacency_mccg(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adjacency = torch.zeros(
        (num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device
    )
    if edge_index.numel() > 0:
        adjacency[edge_index[0], edge_index[1]] = 1.0
    adjacency += torch.eye(num_nodes, device=edge_index.device)
    return adjacency


def diffusion_mccg(adjacency: torch.Tensor, steps: int = 2) -> torch.Tensor:
    transition = F.normalize(adjacency, p=1, dim=0)
    diffusion = torch.zeros_like(transition)
    for power in range(1, steps + 1):
        diffusion += torch.linalg.matrix_power(transition, power)
    return diffusion / steps


def degree_mccg(indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    result = torch.zeros(num_nodes, dtype=torch.float32, device=indices.device)
    if indices.numel() > 0:
        result.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
    return result


def pagerank_mccg(
    edge_index: torch.Tensor, num_nodes: int, damping: float = 0.85, steps: int = 10
) -> torch.Tensor:
    out_degree = degree_mccg(edge_index[0], num_nodes).clamp_min(1.0)
    scores = torch.ones(num_nodes, dtype=torch.float32, device=edge_index.device)
    for _ in range(steps):
        messages = scores[edge_index[0]] / out_degree[edge_index[0]]
        aggregate = torch.zeros_like(scores)
        if messages.numel() > 0:
            aggregate.index_add_(0, edge_index[1], messages)
        scores = (1 - damping) * scores + damping * aggregate
    return scores


def eigenvector_centrality_mccg(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.t().detach().cpu().tolist())
    try:
        values = nx.eigenvector_centrality_numpy(graph)
        centrality = [values[index] for index in range(num_nodes)]
    except (nx.NetworkXException, TypeError):
        centrality = [1.0] * num_nodes
    return torch.tensor(centrality, dtype=torch.float32, device=edge_index.device)


def _inverse_importance_mccg(values: torch.Tensor) -> torch.Tensor:
    denominator = values.max() - values.mean()
    if not torch.isfinite(denominator) or denominator.abs() < 1e-12:
        return torch.ones_like(values)
    result = (values.max() - values) / denominator
    return torch.nan_to_num(result, nan=1.0, posinf=1.0, neginf=0.0)


def edge_drop_weights_mccg(
    edge_index: torch.Tensor, num_nodes: int, scheme: str
) -> torch.Tensor:
    if edge_index.shape[1] == 0:
        return torch.empty(0, dtype=torch.float32, device=edge_index.device)
    if scheme == "degree":
        centrality = degree_mccg(edge_index[1], num_nodes)
    elif scheme == "pr":
        centrality = pagerank_mccg(edge_index, num_nodes, steps=200)
    elif scheme == "evc":
        centrality = eigenvector_centrality_mccg(edge_index, num_nodes).clamp_min(1e-8)
    else:
        raise ValueError(f"Undefined MCCG drop scheme: {scheme}")
    return _inverse_importance_mccg(centrality[edge_index[1]].clamp_min(1e-12).log())


def feature_drop_weights_mccg(
    features: torch.Tensor, node_centrality: torch.Tensor
) -> torch.Tensor:
    importance = features.abs().t() @ node_centrality
    return _inverse_importance_mccg(importance.clamp_min(1e-12).log())


def _drop_probabilities_mccg(
    weights: torch.Tensor, probability: float, threshold: float
) -> torch.Tensor:
    if weights.numel() == 0:
        return weights
    mean = weights.mean()
    if not torch.isfinite(mean) or mean.abs() < 1e-12:
        scaled = torch.full_like(weights, probability)
    else:
        scaled = weights / mean * probability
    return torch.nan_to_num(scaled, nan=probability).clamp(min=0.0, max=threshold)


def drop_edges_weighted_mccg(
    edge_index: torch.Tensor,
    weights: torch.Tensor,
    probability: float,
    threshold: float = 0.7,
) -> torch.Tensor:
    probabilities = _drop_probabilities_mccg(weights, probability, threshold)
    keep = torch.bernoulli(1.0 - probabilities).bool()
    return edge_index[:, keep]


def drop_features_weighted_mccg(
    features: torch.Tensor,
    weights: torch.Tensor,
    probability: float,
    threshold: float = 0.7,
) -> torch.Tensor:
    probabilities = _drop_probabilities_mccg(weights, probability, threshold)
    drop = torch.bernoulli(probabilities).bool()
    augmented = features.clone()
    augmented[:, drop] = 0.0
    return augmented


def centralities_mccg(
    edge_index: torch.Tensor,
    features: torch.Tensor,
    num_nodes: int,
    scheme: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_weights = edge_drop_weights_mccg(edge_index, num_nodes, scheme)
    if scheme == "degree":
        node_centrality = degree_mccg(edge_index[1], num_nodes)
    elif scheme == "pr":
        node_centrality = pagerank_mccg(edge_index, num_nodes)
    elif scheme == "evc":
        node_centrality = eigenvector_centrality_mccg(edge_index, num_nodes)
    else:
        raise ValueError(f"Undefined MCCG drop scheme: {scheme}")
    feature_weights = feature_drop_weights_mccg(features, node_centrality)
    return edge_weights, feature_weights
