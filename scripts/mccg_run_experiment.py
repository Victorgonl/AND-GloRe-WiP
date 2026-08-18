import argparse
import os

from andglore.baseline.mccg import TrainingArgsMCCG, run_experiment_mccg
from andglore.utils import load_yaml_config


def main_mccg() -> None:
    parser_mccg = argparse.ArgumentParser(description="Run an MCCG experiment.")
    parser_mccg.add_argument("--config", required=True, help="Path to an MCCG YAML config")
    args_mccg = parser_mccg.parse_args()
    config_mccg = load_yaml_config(args_mccg.config)

    training_config_mccg = dict(config_mccg.get("training_args", {}))
    training_config_mccg["seeds"] = tuple(training_config_mccg["seeds"])
    training_config_mccg["layer_shape"] = tuple(training_config_mccg["layer_shape"])
    training_args_mccg = TrainingArgsMCCG(**training_config_mccg)

    dataset_name_mccg = config_mccg["dataset_name"]
    results_folder_mccg = config_mccg.get("results_folder")
    results_path_mccg = (
        os.path.join(results_folder_mccg, f"results-{dataset_name_mccg}_mccg.csv")
        if results_folder_mccg
        else None
    )
    run_experiment_mccg(
        dataset_name=dataset_name_mccg,
        networks_path=config_mccg["networks_path"],
        training_args=training_args_mccg,
        selected_names=config_mccg.get("selected_names"),
        results_csv_path=results_path_mccg,
        outputs_path=config_mccg.get("outputs_path"),
        log_file_path=config_mccg.get("log_file"),
    )


if __name__ == "__main__":
    main_mccg()
