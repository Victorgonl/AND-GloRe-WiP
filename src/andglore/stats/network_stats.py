from collections import Counter
from typing import Any

import networkx as nx
import pandas as pd


def network_statistics(graph: nx.Graph) -> dict[str, Any]:
    node_counts = Counter(
        data.get("type", "unknown") for _, data in graph.nodes(data=True)
    )
    relation_counts = Counter(
        data.get("relation", "unknown") for _, _, data in graph.edges(data=True)
    )

    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()

    node_statistics = []
    for node_type, count in sorted(node_counts.items()):
        nodes = [
            node
            for node, data in graph.nodes(data=True)
            if data.get("type", "unknown") == node_type
        ]

        degrees = [graph.degree(node) for node in nodes]

        node_statistics.append(
            {
                "node_type": node_type,
                "count": count,
                "percentage": 100 * count / total_nodes if total_nodes else 0.0,
                "mean_degree": sum(degrees) / len(degrees) if degrees else 0.0,
            }
        )

    relation_statistics = [
        {
            "relation": relation,
            "count": count,
            "percentage": 100 * count / total_edges if total_edges else 0.0,
        }
        for relation, count in sorted(relation_counts.items())
    ]

    return {
        "graph_name": graph.name,
        "split": graph.graph.get("split"),
        "dataset": graph.graph.get("dataset"),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "node_statistics": f"\n{pd.DataFrame(node_statistics)}",
        "relation_statistics": pd.DataFrame(relation_statistics),
    }
