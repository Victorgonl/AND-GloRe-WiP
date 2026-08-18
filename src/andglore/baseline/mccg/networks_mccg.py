import ast
import os
import re
from collections import defaultdict
from itertools import combinations, zip_longest
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import torch
from tqdm import tqdm

_RELATION_STOPWORDS = {
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "a",
    "academy",
    "acta",
    "affiliatedtwo",
    "al",
    "an",
    "analysis",
    "and",
    "applied",
    "are",
    "as",
    "at",
    "based",
    "be",
    "beijing",
    "between",
    "by",
    "c",
    "can",
    "cell",
    "center",
    "chengdu",
    "chemical",
    "chemistry",
    "china",
    "chinese",
    "co",
    "college",
    "communications",
    "conference",
    "dalian",
    "data",
    "daxue",
    "department",
    "dept",
    "different",
    "edu",
    "education",
    "engineering",
    "et",
    "for",
    "from",
    "gongcheng",
    "group",
    "guangzhou",
    "h",
    "has",
    "have",
    "high",
    "hospital",
    "ieee",
    "in",
    "information",
    "inst",
    "institute",
    "international",
    "is",
    "it",
    "japan",
    "journal",
    "jishu",
    "key",
    "lab",
    "laboratory",
    "lanzhou",
    "letters",
    "ltd",
    "materials",
    "medical",
    "method",
    "ministrymethod",
    "n",
    "nanjing",
    "national",
    "of",
    "on",
    "or",
    "p",
    "people",
    "peoples",
    "physics",
    "pr",
    "proceedings",
    "r",
    "research",
    "results",
    "s",
    "school",
    "sci",
    "science",
    "sciences",
    "shanghai",
    "sichuan",
    "sinica",
    "society",
    "state",
    "study",
    "system",
    "technology",
    "than",
    "that",
    "the",
    "these",
    "this",
    "tianjing",
    "time",
    "to",
    "univ",
    "universities",
    "university",
    "usa",
    "used",
    "using",
    "was",
    "we",
    "were",
    "which",
    "with",
    "wuhan",
    "xi",
    "xuebao",
    "yu",
    "zhejiang",
}


def _parse_list_mccg(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    return []


def _tokenize_relation_mccg(
    value: Any, *, null_when_filtered: bool = False
) -> set[str]:
    if not isinstance(value, str):
        return set()
    tokens = re.sub(r"[\W_]+", " ", value.lower(), flags=re.UNICODE).split()
    filtered = {
        token for token in tokens if len(token) > 1 and token not in _RELATION_STOPWORDS
    }
    if value.strip() and not filtered and null_when_filtered:
        return {"null"}
    return filtered


def _compact_author_mccg(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalpha())


def _organization_id_mccg(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _filter_coauthors_mccg(
    coauthor_orgs_by_paper: list[list[tuple[str, str]]],
    max_orgs_per_author: Optional[int],
    max_author_paper_ratio: Optional[float],
) -> list[set[str]]:
    authors_to_split: set[str] = set()
    if max_orgs_per_author is not None:
        authors_by_org: dict[str, set[str]] = defaultdict(set)
        for paper_relations in coauthor_orgs_by_paper:
            for author, organization in paper_relations:
                if organization:
                    authors_by_org[organization].add(author)
        for organization_authors in authors_by_org.values():
            if len(organization_authors) > max_orgs_per_author:
                authors_to_split.update(organization_authors)

    coauthors_by_paper: list[set[str]] = []
    for paper_relations in coauthor_orgs_by_paper:
        paper_coauthors: set[str] = set()
        for author, organization in paper_relations:
            if author in authors_to_split and organization:
                paper_coauthors.add(f"{author} [{organization}]")
            else:
                paper_coauthors.add(author)
        coauthors_by_paper.append(paper_coauthors)

    if max_author_paper_ratio is not None:
        occurrence_counts: dict[str, int] = defaultdict(int)
        for paper_coauthors in coauthors_by_paper:
            for author in paper_coauthors:
                occurrence_counts[author] += 1
        ratio_threshold = len(coauthors_by_paper) * max_author_paper_ratio
        authors_to_remove = {
            author
            for author, count in occurrence_counts.items()
            if count > ratio_threshold
        }
        if authors_to_remove:
            coauthors_by_paper = [
                paper_coauthors.difference(authors_to_remove)
                for paper_coauthors in coauthors_by_paper
            ]

    return coauthors_by_paper


def load_preprocessed_mccg(path: str | os.PathLike[str]) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise ValueError("preprocessed_path must point to preprocessed.csv")
    return pd.read_csv(input_path)


def _load_features_mccg(path: str | os.PathLike[str]) -> dict[Any, torch.Tensor]:
    raw_features = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw_features, dict) or not raw_features:
        raise ValueError("features.pt must be a non-empty dictionary keyed by paper id")

    features: dict[Any, torch.Tensor] = {}
    dimensions: set[int] = set()
    for paper_id, value in raw_features.items():
        if not torch.is_tensor(value) or value.ndim != 1:
            raise ValueError("Every features.pt value must be a one-dimensional tensor")
        tensor = value.detach().cpu().float()
        features[paper_id] = tensor
        dimensions.add(tensor.numel())
    if len(dimensions) != 1:
        raise ValueError("All features.pt vectors must have the same dimension")
    return features


def _pair_overlap_counts_mccg(values: list[set[str]]) -> dict[tuple[int, int], int]:
    inverted: dict[str, list[int]] = defaultdict(list)
    for paper_index, entities in enumerate(values):
        for entity in entities:
            inverted[entity].append(paper_index)

    counts: dict[tuple[int, int], int] = defaultdict(int)
    for paper_indices in inverted.values():
        for left, right in combinations(sorted(set(paper_indices)), 2):
            counts[(left, right)] += 1
    return counts


def _feature_for_paper_mccg(
    paper_id: Any,
    features: dict[Any, torch.Tensor],
    string_features: dict[str, torch.Tensor],
    feature_dim: int,
) -> torch.Tensor:
    feature = features.get(paper_id)
    if feature is None:
        feature = string_features.get(str(paper_id))
    if feature is None:
        return torch.zeros(feature_dim, dtype=torch.float32)
    return feature


def _build_name_network_mccg(
    group: pd.DataFrame,
    target_name: str,
    split: str,
    dataset_name: str,
    features: dict[Any, torch.Tensor],
    max_orgs_per_author: Optional[int] = None,
    max_author_paper_ratio: Optional[float] = None,
    max_org_affiliation: Optional[int] = None,
) -> dict[str, Any]:
    group = group.drop_duplicates(subset="id", keep="first")
    paper_ids = group["id"].tolist()
    labels = torch.tensor(
        group["label"].astype(int).to_numpy(copy=True), dtype=torch.long
    )

    feature_dim = next(iter(features.values())).numel()
    string_features = {str(key): value for key, value in features.items()}
    paper_features = torch.stack(
        [
            _feature_for_paper_mccg(paper_id, features, string_features, feature_dim)
            for paper_id in paper_ids
        ]
    )

    coauthor_orgs_by_paper: list[list[tuple[str, str]]] = []
    target_org_tokens_by_paper: list[set[str]] = []
    venue_tokens_by_paper: list[set[str]] = []
    compact_target_name = _compact_author_mccg(target_name)

    if max_org_affiliation is not None and max_org_affiliation < 0:
        raise ValueError("max_org_affiliation must be non-negative")

    affiliations_by_author: dict[str, set[str]] = defaultdict(set)
    if max_org_affiliation is not None:
        for row in group.itertuples(index=False):
            authors = [str(author) for author in _parse_list_mccg(row.authors)]
            orgs = _parse_list_mccg(row.orgs)
            for author, org in zip(authors, orgs):
                compact_author = _compact_author_mccg(author)
                organization = _organization_id_mccg(org)
                if (
                    compact_author
                    and compact_author != compact_target_name
                    and organization
                ):
                    affiliations_by_author[compact_author].add(organization)

    authors_over_affiliation_limit = {
        author
        for author, organizations in affiliations_by_author.items()
        if len(organizations) > max_org_affiliation
    } if max_org_affiliation is not None else set()

    for row in group.itertuples(index=False):
        authors = [str(author) for author in _parse_list_mccg(row.authors)]
        orgs = _parse_list_mccg(row.orgs)
        coauthor_orgs: list[tuple[str, str]] = []
        target_org_tokens: set[str] = set()
        for author, org in zip_longest(authors, orgs, fillvalue=""):
            compact_author = _compact_author_mccg(author)
            if compact_author and compact_author != compact_target_name:
                if compact_author in authors_over_affiliation_limit:
                    target_org_tokens.update(_tokenize_relation_mccg(org))
                    continue
                coauthor_orgs.append((compact_author, _organization_id_mccg(org)))
        coauthor_orgs_by_paper.append(coauthor_orgs)

        for author, org in zip(authors, orgs):
            if _compact_author_mccg(author) == compact_target_name:
                target_org_tokens.update(_tokenize_relation_mccg(org))
        target_org_tokens_by_paper.append(target_org_tokens)
        venue_tokens_by_paper.append(
            _tokenize_relation_mccg(row.venue, null_when_filtered=True)
        )

    coauthors_by_paper = _filter_coauthors_mccg(
        coauthor_orgs_by_paper,
        max_orgs_per_author=max_orgs_per_author,
        max_author_paper_ratio=max_author_paper_ratio,
    )
    author_counts = _pair_overlap_counts_mccg(coauthors_by_paper)
    org_counts = _pair_overlap_counts_mccg(target_org_tokens_by_paper)
    venue_counts = _pair_overlap_counts_mccg(venue_tokens_by_paper)

    candidate_pairs = sorted(set(author_counts) | set(org_counts))
    if candidate_pairs:
        pair_index = torch.tensor(candidate_pairs, dtype=torch.long).t().contiguous()
        author_overlap = torch.tensor(
            [author_counts.get(pair, 0) for pair in candidate_pairs], dtype=torch.long
        )
        org_jaccard_values: list[float] = []
        for pair in candidate_pairs:
            shared = org_counts.get(pair, 0)
            union = (
                len(target_org_tokens_by_paper[pair[0]])
                + len(target_org_tokens_by_paper[pair[1]])
                - shared
            )
            org_jaccard_values.append(shared / union if union else 0.0)
        org_jaccard = torch.tensor(org_jaccard_values, dtype=torch.float32)
        venue_overlap = torch.tensor(
            [venue_counts.get(pair, 0) for pair in candidate_pairs], dtype=torch.long
        )
    else:
        pair_index = torch.empty((2, 0), dtype=torch.long)
        author_overlap = torch.empty(0, dtype=torch.long)
        org_jaccard = torch.empty(0, dtype=torch.float32)
        venue_overlap = torch.empty(0, dtype=torch.long)

    return {
        "name": target_name,
        "split": split,
        "dataset": dataset_name,
        "paper_ids": paper_ids,
        "labels": labels,
        "features": paper_features,
        "pair_index": pair_index,
        "author_overlap": author_overlap,
        "org_jaccard": org_jaccard,
        "venue_overlap": venue_overlap,
        "max_orgs_per_author": max_orgs_per_author,
        "max_author_paper_ratio": max_author_paper_ratio,
        "max_org_affiliation": max_org_affiliation,
    }


def build_homogeneous_networks_mccg(
    preprocessed_path: str | os.PathLike[str],
    features_path: str | os.PathLike[str],
    dataset_name: str,
    save_folder: Optional[str | os.PathLike[str]] = None,
    selected_names: Optional[list[str]] = None,
    splits: Optional[list[str]] = None,
    logs_file: Optional[str | os.PathLike[str]] = None,
    max_orgs_per_author: Optional[int] = None,
    max_author_paper_ratio: Optional[float] = None,
    max_org_affiliation: Optional[int] = None,
    networks_path: Optional[str | os.PathLike[str]] = None,
) -> list[dict[str, Any]]:
    dataframe = load_preprocessed_mccg(preprocessed_path)
    required = {"id", "name", "split", "label", "authors", "orgs", "venue"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if selected_names is not None:
        dataframe = dataframe[dataframe["name"].isin(selected_names)]
    if splits is not None:
        dataframe = dataframe[dataframe["split"].isin(splits)]
    if dataframe.empty:
        raise ValueError("No publications remain after applying name/split filters")

    features = _load_features_mccg(features_path)
    networks: list[dict[str, Any]] = []
    grouped = dataframe.groupby("name", sort=False)
    for target_name, group in tqdm(
        grouped, desc="Generating MCCG homogeneous networks", unit="name"
    ):
        split_values = group["split"].dropna().unique()
        if len(split_values) != 1:
            raise ValueError(f"Name group {target_name!r} spans multiple splits")
        networks.append(
            _build_name_network_mccg(
                group=group,
                target_name=str(target_name),
                split=str(split_values[0]),
                dataset_name=dataset_name,
                features=features,
                max_orgs_per_author=max_orgs_per_author,
                max_author_paper_ratio=max_author_paper_ratio,
                max_org_affiliation=max_org_affiliation,
            )
        )

    if networks_path is not None:
        output_path = Path(networks_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(networks, output_path)
    elif save_folder is not None:
        output_folder = Path(save_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        output_splits: Iterable[str] = splits or dict.fromkeys(
            network["split"] for network in networks
        )
        for split in output_splits:
            split_networks = [
                network for network in networks if network["split"] == split
            ]
            torch.save(split_networks, output_folder / f"networks_{split}_mccg.pt")

    if logs_file is not None:
        log_path = Path(logs_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"Generated {len(networks)} MCCG homogeneous networks for {dataset_name}.\n"
            f"max_orgs_per_author={max_orgs_per_author}\n"
            f"max_author_paper_ratio={max_author_paper_ratio}\n"
            f"max_org_affiliation={max_org_affiliation}\n",
            encoding="utf-8",
        )
    return networks
