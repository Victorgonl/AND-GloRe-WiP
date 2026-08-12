from typing import Optional

from andglore.networks import build_ambiguous_networks
from andglore.utils import load_yaml_config, parse_args


def generate_networks(
    dataset_name: str,
    selected_names: Optional[list[str]],
    splits: list[str],
    max_orgs_per_author: Optional[int] = None,
    max_author_paper_ratio: Optional[float] = None,
):
    """
    Generate AND Networks using preprocessed.csv and features.pt.
    """

    build_ambiguous_networks(
        dataset_name=dataset_name,
        df_path=f"data/{dataset_name}/preprocessed.csv",
        save_folder=f"data/{dataset_name}/",
        logs_file=f"logs/networks-{dataset_name}.log",
        embeddings_path=f"data/{dataset_name}/features.pt",
        splits=splits,
        selected_names=selected_names,
        max_orgs_per_author=max_orgs_per_author,
        max_author_paper_ratio=max_author_paper_ratio,
    )


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    generate_networks(**config)


if __name__ == "__main__":
    main()
