from typing import Optional

from andglore.networks import build_ambiguous_networks

from andglore.utils import load_yaml_config, parse_args


def generate_networks(
    dataset_name: str,
    selected_names: Optional[list[str]],
    splits: list[str],
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
    )


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset_name = config["dataset_name"]
    selected_names = config["selected_names"]
    splits = config["splits"]

    generate_networks(
        dataset_name=dataset_name,
        selected_names=selected_names,
        splits=splits,
    )


if __name__ == "__main__":
    main()
