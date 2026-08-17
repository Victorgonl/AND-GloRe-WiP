import argparse

from andglore.baseline.mccg import build_homogeneous_networks_mccg
from andglore.utils import load_yaml_config


def main_mccg() -> None:
    parser_mccg = argparse.ArgumentParser(
        description="Generate MCCG networks from preprocessed.csv and features.pt."
    )
    parser_mccg.add_argument(
        "--config", required=True, help="Path to an MCCG YAML config"
    )
    parser_mccg.add_argument(
        "--max_orgs_per_author",
        type=int,
        default=None,
        help="Maximum authors sharing an organization before coauthors are split by organization.",
    )
    parser_mccg.add_argument(
        "--max_author_paper_ratio",
        type=float,
        default=None,
        help="Remove coauthors occurring in more than this fraction of a name group's papers.",
    )
    args_mccg = parser_mccg.parse_args()
    config_mccg = load_yaml_config(args_mccg.config)
    dataset_name_mccg = config_mccg["dataset_name"]
    build_homogeneous_networks_mccg(
        preprocessed_path=config_mccg.get(
            "preprocessed_path", f"data/{dataset_name_mccg}/preprocessed.csv"
        ),
        features_path=config_mccg.get(
            "features_path", f"data/{dataset_name_mccg}/features.pt"
        ),
        dataset_name=dataset_name_mccg,
        save_folder=config_mccg.get("networks_folder", f"data/{dataset_name_mccg}"),
        selected_names=config_mccg.get("selected_names"),
        splits=config_mccg.get("splits"),
        logs_file=config_mccg.get(
            "networks_log_file", f"logs/networks-{dataset_name_mccg}_mccg.log"
        ),
        max_orgs_per_author=(
            args_mccg.max_orgs_per_author
            if args_mccg.max_orgs_per_author is not None
            else config_mccg.get("max_orgs_per_author")
        ),
        max_author_paper_ratio=(
            args_mccg.max_author_paper_ratio
            if args_mccg.max_author_paper_ratio is not None
            else config_mccg.get("max_author_paper_ratio")
        ),
    )


if __name__ == "__main__":
    main_mccg()
