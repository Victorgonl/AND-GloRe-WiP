import os

import torch

from andglore.trainer import TrainingArgs, run_andglore_experiment
from andglore.utils import load_yaml_config, parse_args


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset_name = config["dataset_name"]
    networks_path = config["networks_path"]
    results_folder = config.get("results_folder")
    outputs_path = config.get("outputs_folder")

    training_cfg = dict(config.get("training_args", {}))
    if "seeds" in training_cfg:
        training_cfg["seeds"] = tuple(training_cfg["seeds"])
    training_args = TrainingArgs(**training_cfg)

    # Determine results path
    results_path = None
    if results_folder:
        results_path = os.path.join(results_folder, f"results-{dataset_name}.csv")

    # Determine log file
    log_file = config.get("log_file")
    if log_file:
        log_file = log_file

    selected_names = config.get("selected_names")

    networks = torch.load(
        networks_path,
        weights_only=False,
    )

    run_andglore_experiment(
        dataset_name=dataset_name,
        networks=networks,
        training_args=training_args,
        selected_names=selected_names,
        results_csv_path=results_path,  # type: ignore
        outputs_path=outputs_path,  # type: ignore
        log_file_path=log_file,  # type: ignore
    )


if __name__ == "__main__":
    main()
