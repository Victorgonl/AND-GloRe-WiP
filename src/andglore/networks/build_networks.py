import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import networkx as nx
import pandas as pd
import torch
from tqdm import tqdm

MIN_AUTHOR_NODES = 2
MIN_ORG_NODES = 2
MIN_VENUE_NODES = 2
MAX_AUTHORS_OR_ORGS_FOR_NON_TEST = 10
NODE_TYPES = Literal["paper", "author", "venue", "org"]


def load_dataset(path: str | os.PathLike[str]) -> pd.DataFrame:
    return pd.read_csv(path)


def load_embeddings(path: str | os.PathLike[str]) -> dict[Any, torch.Tensor]:
    return _validate_embeddings(torch.load(path, map_location="cpu", weights_only=True))


def _parse_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def _venue_id(venue: str, shared_names: set[str]) -> str:
    return f"{venue} [venue]" if venue in shared_names else venue


def _org_id(org: str, shared_names: set[str]) -> str:
    return f"{org} [org]" if org in shared_names else org


def _add_edge(
    graph: nx.Graph,
    source: Any,
    target: Any,
    relation: str,
) -> None:
    graph.add_edge(source, target, relation=relation)


def _nodes_of_type(graph: nx.Graph, node_type: str) -> list[Any]:
    return [node for node, data in graph.nodes(data=True) if data["type"] == node_type]


def _ensure_min_nodes(
    graph: nx.Graph,
    paper_ids: list[Any],
    node_type: str,
    minimum: int,
    relation: str,
) -> None:
    missing = minimum - len(_nodes_of_type(graph, node_type))
    for index in range(max(0, missing)):
        node_id = f"mock_{node_type}_{index}"
        graph.add_node(node_id, type=node_type)
        if paper_ids:
            _add_edge(graph, paper_ids[0], node_id, relation)


def _validate_embeddings(value: Any) -> dict[Any, torch.Tensor]:
    if not isinstance(value, dict) or not value:
        raise ValueError("features.pt must be a non-empty dictionary keyed by paper id")

    result: dict[Any, torch.Tensor] = {}
    dimensions: set[int] = set()
    for paper_id, embedding in value.items():
        if not torch.is_tensor(embedding) or embedding.ndim != 1:
            raise ValueError(
                "Each publication feature must be a one-dimensional tensor"
            )
        embedding = embedding.detach().cpu().float()
        result[paper_id] = embedding
        dimensions.add(embedding.numel())
    if len(dimensions) != 1:
        raise ValueError("All publication features must have the same dimension")
    return result


def _remove_singular_degree_non_paper(graph: nx.Graph) -> None:
    """
    Iteratively removes non-paper nodes (authors, orgs, venues)
    that have a degree of 1 or less until the graph stabilizes.
    """
    while True:
        # Identify non-paper nodes with degree <= 1
        nodes_to_remove = [
            node
            for node, data in graph.nodes(data=True)
            if data.get("type") != "paper" and graph.degree(node) <= 1
        ]

        # If no nodes meet the criteria, the pruning is complete
        if not nodes_to_remove:
            break

        # Remove the identified nodes (which automatically removes their edges)
        graph.remove_nodes_from(nodes_to_remove)


def _build_network(
    group: pd.DataFrame,
    target_name: str,
    split: str,
    dataset_name: str,
    embeddings: dict[Any, torch.Tensor],
    max_orgs_per_author: Optional[int] = None,
    max_author_paper_ratio: Optional[float] = None,
    ignore_authors: Optional[list[str]] = None,
    max_org_affiliation: Optional[int] = None,
) -> nx.Graph:
    graph = nx.Graph(name=target_name, split=split, dataset=dataset_name)
    papers: list[tuple[Any, int, int]] = []
    author_org: dict[str, set[str]] = defaultdict(set)
    paper_org: dict[Any, set[str]] = defaultdict(set)
    paper_venue: dict[Any, set[str]] = defaultdict(set)
    paper_author: dict[Any, set[str]] = defaultdict(set)

    # Track exact (author, org) pairings specific to each paper
    paper_author_org: dict[Any, set[tuple[str, str]]] = defaultdict(set)

    venue_occurrences: dict[str, int] = defaultdict(int)
    author_occurrences: dict[str, int] = defaultdict(int)
    org_occurrences: dict[str, int] = defaultdict(int)
    ignored_authors = set(ignore_authors or [])
    if max_org_affiliation is not None:
        if max_org_affiliation < 0:
            raise ValueError("max_org_affiliation must be non-negative")

        affiliations_by_author: dict[str, set[str]] = defaultdict(set)
        for _, row in group.iterrows():
            authors = _parse_list(row["authors"])
            orgs = _parse_list(row["orgs"])
            if split != "test":
                authors = authors[:MAX_AUTHORS_OR_ORGS_FOR_NON_TEST]
                orgs = orgs[:MAX_AUTHORS_OR_ORGS_FOR_NON_TEST]
            for author, org in zip(authors, orgs):
                if author and author != target_name and org:
                    affiliations_by_author[author].add(org)

        ignored_authors.update(
            author
            for author, organizations in affiliations_by_author.items()
            if len(organizations) > max_org_affiliation
        )

    for _, row in group.iterrows():
        paper_id = row["id"]
        venue = row["venue"] if pd.notna(row["venue"]) else ""
        authors = _parse_list(row["authors"])
        orgs = _parse_list(row["orgs"])
        label = int(row["label"])
        year = int(row["year"]) if pd.notna(row["year"]) else -1
        papers.append((paper_id, label, year))

        if venue:
            venue_occurrences[venue] += 1
            paper_venue[paper_id].add(venue)

        if split != "test":
            authors = authors[:MAX_AUTHORS_OR_ORGS_FOR_NON_TEST]
            orgs = orgs[:MAX_AUTHORS_OR_ORGS_FOR_NON_TEST]

        for author, org in zip(authors, orgs):
            if author and author != target_name:
                if author in ignored_authors:
                    if org:
                        paper_org[paper_id].add(org)
                        org_occurrences[org] += 1
                    continue
                author_occurrences[author] += 1
                paper_author[paper_id].add(author)
                if org:
                    author_org[author].add(org)
                    org_occurrences[org] += 1
                    paper_author_org[paper_id].add((author, org))
            elif org and author == target_name:
                paper_org[paper_id].add(org)
                org_occurrences[org] += 1

    # Check for name collisions between venues and orgs
    shared_names = set(venue_occurrences.keys()) & set(org_occurrences.keys())

    embedding_dim = next(iter(embeddings.values())).numel()
    papers = list(dict.fromkeys(papers))
    for paper_id, label, year in papers:
        feature = embeddings.get(paper_id)
        if feature is None:
            feature = torch.zeros(embedding_dim, dtype=torch.float32)
        graph.add_node(
            paper_id,
            type="paper",
            label=label,
            year=year,
            feat=feature.detach().cpu().float(),
        )

        for venue in sorted(paper_venue[paper_id]):
            if venue_occurrences[venue] >= 1:
                venue_node = _venue_id(venue, shared_names)
                graph.add_node(venue_node, type="venue", name=venue)
                _add_edge(graph, paper_id, venue_node, "published_in")

        for author in sorted(paper_author[paper_id]):
            if author_occurrences[author] >= 1:
                graph.add_node(author, type="author")
                _add_edge(graph, paper_id, author, "written_by")
                for org in sorted(author_org[author]):
                    if org_occurrences[org] >= 1:
                        org_node = _org_id(org, shared_names)
                        graph.add_node(org_node, type="org", name=org)
                        _add_edge(graph, author, org_node, "affiliated_with")

        for org in sorted(paper_org[paper_id]):
            if org_occurrences[org] >= 1:
                org_node = _org_id(org, shared_names)
                graph.add_node(org_node, type="org", name=org)
                _add_edge(graph, paper_id, org_node, "author_affiliated_with")

    # Apply max_orgs_per_author filtering
    if max_orgs_per_author is not None:
        authors_to_remove = set()

        # 1. Find authors tied to an org exceeding the limit
        for org_node in _nodes_of_type(graph, "org"):
            org_authors = [
                neighbor
                for neighbor in graph.neighbors(org_node)
                if graph.nodes[neighbor].get("type") == "author"
            ]
            if len(org_authors) > max_orgs_per_author:
                authors_to_remove.update(org_authors)

        edges_to_add = []

        # 2. Map precisely which author and org belong to which paper
        for author in authors_to_remove:
            author_papers = [
                neighbor
                for neighbor in graph.neighbors(author)
                if graph.nodes[neighbor].get("type") == "paper"
            ]

            for paper in author_papers:
                # Find the orgs this author was specifically affiliated with for *this* paper
                orgs_for_this_paper = [
                    o for (a, o) in paper_author_org[paper] if a == author
                ]

                if not orgs_for_this_paper:
                    # If they had no org on this paper, just reconnect the standard author node
                    edges_to_add.append((paper, author))
                else:
                    for org in orgs_for_this_paper:
                        org_node = _org_id(org, shared_names)
                        # Ensure we get the clean name to format nicely
                        org_name = graph.nodes.get(org_node, {}).get("name", org)
                        new_author_node = f"{author} [{org_name}]"
                        edges_to_add.append((paper, new_author_node))

        # 3. Remove the identified original authors
        if authors_to_remove:
            graph.remove_nodes_from(authors_to_remove)

        # 4. Add the combined author nodes back (or original ones if no org was present)
        for paper, node_id in edges_to_add:
            if not graph.has_node(node_id):
                graph.add_node(node_id, type="author")
            _add_edge(graph, paper, node_id, "written_by")

        # 5. Clean up any orgs that now have a degree of 0
        orgs_to_remove = [
            node for node in _nodes_of_type(graph, "org") if graph.degree(node) == 0
        ]
        if orgs_to_remove:
            graph.remove_nodes_from(orgs_to_remove)

    # Apply max_author_paper_ratio filtering
    if max_author_paper_ratio is not None:
        total_papers = len(papers)
        ratio_threshold = total_papers * max_author_paper_ratio

        authors_to_drop = [
            author_node
            for author_node in _nodes_of_type(graph, "author")
            if sum(
                1
                for neighbor in graph.neighbors(author_node)
                if graph.nodes[neighbor].get("type") == "paper"
            )
            > ratio_threshold
        ]

        if authors_to_drop:
            graph.remove_nodes_from(authors_to_drop)

    paper_ids = [paper_id for paper_id, _, _ in papers]

    _remove_singular_degree_non_paper(graph)

    # Ensure minimum node counts
    _ensure_min_nodes(graph, paper_ids, "author", MIN_AUTHOR_NODES, "written_by")
    _ensure_min_nodes(graph, paper_ids, "org", MIN_ORG_NODES, "author_affiliated_with")
    _ensure_min_nodes(graph, paper_ids, "venue", MIN_VENUE_NODES, "published_in")

    # Graph metadata
    graph.name = target_name

    return graph


def build_ambiguous_networks(
    dataset_name: str,
    preprocessed: pd.DataFrame,
    embeddings: Dict[Any, torch.Tensor],
    save_folder: Optional[str] = None,
    logs_file: Optional[str] = None,
    selected_names: Optional[list[str]] = None,
    splits: Optional[list[str]] = None,
    max_orgs_per_author: Optional[int] = None,
    max_author_paper_ratio: Optional[float] = None,
    ignore_authors: Optional[list[str]] = None,
    max_org_affiliation: Optional[int] = None,
    networks_path: Optional[str | os.PathLike[str]] = None,
) -> list[nx.Graph]:
    """Generate and save one undirected NetworkX graph per ambiguous name."""

    dataframe = preprocessed
    required = {"id", "name", "split", "label", "authors", "orgs", "venue", "year"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if selected_names is not None:
        dataframe = dataframe[dataframe["name"].isin(selected_names)]
    if splits is not None:
        dataframe = dataframe[dataframe["split"].isin(splits)]

    networks: list[nx.Graph] = []
    grouped = dataframe.groupby("name", sort=False)
    for target_name, group in tqdm(
        grouped, desc="Generating ambiguous networks", unit="name"
    ):
        split_values = group["split"].dropna().unique()
        if len(split_values) != 1:
            raise ValueError(f"Name group {target_name!r} spans multiple splits")
        networks.append(
            _build_network(
                group,
                str(target_name),
                str(split_values[0]),
                dataset_name,
                embeddings,
                max_orgs_per_author=max_orgs_per_author,
                max_author_paper_ratio=max_author_paper_ratio,
                ignore_authors=ignore_authors,
                max_org_affiliation=max_org_affiliation,
            )
        )

    if networks_path is not None:
        output_path = Path(networks_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(networks, output_path)
    elif save_folder:
        output_folder = Path(save_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        output_splits: Iterable[str] = splits or dict.fromkeys(
            graph.graph["split"] for graph in networks
        )
        for split in output_splits:
            split_networks = [
                graph for graph in networks if graph.graph["split"] == split
            ]
            torch.save(split_networks, output_folder / f"networks_{split}.pt")

    if logs_file:
        log_path = Path(logs_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"Generated {len(networks)} NetworkX graphs for {dataset_name}.\n"
            f"max_org_affiliation={max_org_affiliation}\n",
            encoding="utf-8",
        )
    return networks
