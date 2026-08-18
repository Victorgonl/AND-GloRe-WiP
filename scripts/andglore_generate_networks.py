from typing import Optional

import torch

from andglore.networks import build_ambiguous_networks
from andglore.networks.build_networks import load_dataset, load_embeddings
from andglore.utils import load_yaml_config, parse_args


def generate_networks(
    dataset_name: str,
    selected_names: Optional[list[str]],
    splits: list[str],
    networks_path: str,
    networks_log_file: Optional[str] = None,
    max_orgs_per_author: Optional[int] = None,
    max_author_paper_ratio: Optional[float] = None,
    ignore_authors: Optional[list[str]] = None,
    max_org_affiliation: Optional[int] = None,
):
    """
    Generate AND Networks using preprocessed.csv and features.pt.
    """

    preprocessed = load_dataset(f"data/{dataset_name}/preprocessed.csv")
    embeddings = load_embeddings(f"data/{dataset_name}/features.pt")

    build_ambiguous_networks(
        dataset_name=dataset_name,
        preprocessed=preprocessed,
        networks_path=networks_path,
        logs_file=networks_log_file,
        embeddings=embeddings,
        splits=splits,
        selected_names=selected_names,
        max_orgs_per_author=max_orgs_per_author,
        max_author_paper_ratio=max_author_paper_ratio,
        ignore_authors=ignore_authors,
        max_org_affiliation=max_org_affiliation,
    )


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)

    generate_networks(
        dataset_name=config["dataset_name"],
        selected_names=config["selected_names"],
        splits=config["splits"],
        networks_path=config["networks_path"],
        networks_log_file=config.get(
            "networks_log_file", f"logs/networks-{config['dataset_name']}.log"
        ),
        max_orgs_per_author=args.max_orgs_per_author,
        max_author_paper_ratio=args.max_author_paper_ratio,
        ignore_authors=config.get("ignore_authors"),
        max_org_affiliation=(
            args.max_org_affiliation
            if args.max_org_affiliation is not None
            else config.get("max_org_affiliation")
        ),
    )
