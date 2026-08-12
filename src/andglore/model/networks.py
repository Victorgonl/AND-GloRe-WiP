from typing import Any, Iterable

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from andglore.networks import NODE_TYPES


def get_nodes(
    graph: nx.Graph,
    node_type: NODE_TYPES,
) -> list[Any]:
    """Return all graph nodes of the requested type."""

    return [
        node for node, data in graph.nodes(data=True) if data.get("type") == node_type
    ]


def get_paper_features(
    graph: nx.Graph,
    paper_nodes: Iterable[Any],
) -> torch.Tensor:
    """Stack paper feature vectors following the given node order."""

    paper_nodes = list(paper_nodes)
    if not paper_nodes:
        raise ValueError("Cannot extract features: the graph has no paper nodes")

    features: list[torch.Tensor] = []

    for paper_node in paper_nodes:
        node_data = graph.nodes[paper_node]

        if node_data.get("type") != "paper":
            raise ValueError(f"Node {paper_node!r} is not a paper node")

        feature = node_data.get("feat")
        if not torch.is_tensor(feature):
            raise ValueError(f"Paper node {paper_node!r} is missing a tensor feature")

        if feature.ndim != 1:  # type: ignore
            raise ValueError(
                f"Feature for paper {paper_node!r} must be one-dimensional"
            )

        features.append(feature.detach().cpu().float())  # type: ignore

    feature_dimensions = {feature.numel() for feature in features}
    if len(feature_dimensions) != 1:
        raise ValueError("All paper features must have the same dimension")

    return torch.stack(features)


def get_paper_labels(
    graph: nx.Graph,
    paper_nodes: Iterable[Any],
) -> torch.Tensor:
    """Extract paper labels following the given node order."""

    labels: list[float] = []

    for paper_node in paper_nodes:
        node_data = graph.nodes[paper_node]

        if node_data.get("type") != "paper":
            raise ValueError(f"Node {paper_node!r} is not a paper node")

        if "label" not in node_data:
            raise ValueError(f"Paper node {paper_node!r} has no label")

        labels.append(float(node_data["label"]))

    return torch.tensor(labels, dtype=torch.float32)


def create_bipartite_matrix(
    graph: nx.Graph,
    relation: str,
    source_nodes: Iterable[Any],
    destination_nodes: Iterable[Any],
) -> sp.coo_matrix:
    """
    Create a source-by-destination sparse matrix for one relation.

    The graph may be undirected, so edge endpoint order is inferred from the
    provided source and destination node collections.
    """

    source_nodes = list(source_nodes)
    destination_nodes = list(destination_nodes)

    source_map = {node: index for index, node in enumerate(source_nodes)}
    destination_map = {node: index for index, node in enumerate(destination_nodes)}

    source_ids: list[int] = []
    destination_ids: list[int] = []

    for first_node, second_node, edge_data in graph.edges(data=True):
        if edge_data.get("relation") != relation:
            continue

        # Orientation already matches source -> destination.
        if first_node in source_map and second_node in destination_map:
            source_ids.append(source_map[first_node])
            destination_ids.append(destination_map[second_node])
            continue

        # Undirected NetworkX graph returned the endpoints in reverse order.
        if second_node in source_map and first_node in destination_map:
            source_ids.append(source_map[second_node])
            destination_ids.append(destination_map[first_node])
            continue

        raise ValueError(
            f"Relation {relation!r} contains edge "
            f"({first_node!r}, {second_node!r}) whose endpoints do not belong "
            "to the supplied source and destination node collections"
        )

    edge_weights = np.ones(len(source_ids), dtype=np.float32)

    return sp.coo_matrix(
        (
            edge_weights,
            (
                np.asarray(source_ids, dtype=np.int64),
                np.asarray(destination_ids, dtype=np.int64),
            ),
        ),
        shape=(len(source_nodes), len(destination_nodes)),
        dtype=np.float32,
    )
