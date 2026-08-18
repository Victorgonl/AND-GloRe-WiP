import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from andglore.baseline.mccg import (
    TrainingArgsMCCG,
    build_homogeneous_networks_mccg,
    run_experiment_mccg,
)
from andglore.networks import build_ambiguous_networks
from andglore.networks.build_networks import load_dataset, load_embeddings
from andglore.trainer import TrainingArgs, run_andglore_experiment
from andglore.utils import AVERAGE_NAME, load_yaml_config

SWEEP_VALUES: tuple[int | None, ...] = (1, 2, 3, 5, 10, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep max_org_affiliation for AND-GloRe and MCCG."
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name, e.g. lagosandv1"
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Unique ID used for the directory under --experiments-root",
    )
    parser.add_argument(
        "--selected-names",
        nargs="+",
        help="Small set of ambiguous names for a debug run (overrides both configs)",
    )
    parser.add_argument(
        "--andglore-config",
        type=Path,
        help="Defaults to configs/andglore_<dataset>.yaml",
    )
    parser.add_argument(
        "--mccg-config",
        type=Path,
        help="Defaults to configs/mccg_<dataset>.yaml",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments/max_org_affiliation_sweep"),
    )
    return parser.parse_args()


def checked_experiment_dir(root: Path, experiment_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id):
        raise ValueError(
            "Experiment ID may contain only letters, numbers, '.', '_' and '-'"
        )

    experiment_dir = root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def training_args(config: dict[str, Any]) -> TrainingArgs:
    values = dict(config.get("training_args", {}))
    if "seeds" in values:
        values["seeds"] = tuple(values["seeds"])
    return TrainingArgs(**values)


def training_args_mccg(config: dict[str, Any]) -> TrainingArgsMCCG:
    values = dict(config.get("training_args", {}))
    if "seeds" in values:
        values["seeds"] = tuple(values["seeds"])
    if "layer_shape" in values:
        values["layer_shape"] = tuple(values["layer_shape"])
    return TrainingArgsMCCG(**values)


def average_pf1(results_path: Path) -> float:
    frame = pd.read_csv(results_path)
    average = frame.loc[frame["name"] == AVERAGE_NAME]
    if len(average) != 1:
        raise ValueError(f"Expected one {AVERAGE_NAME!r} row in {results_path}")
    return float(average.iloc[0]["pF1_mean"])


def plot_results(summary: pd.DataFrame, output_path: Path, dataset: str) -> None:
    labels = [str(value) for value in (1, 2, 3, 5, 10)] + ["Sem Restrição"]
    positions = range(len(labels))

    figure, axis = plt.subplots(figsize=(8, 5))

    for method, marker in (("AND-GloRe", "o"), ("MCCG", "s")):
        method_rows = summary.loc[summary["method"] == method].reset_index(drop=True)

        if len(method_rows) != len(labels):
            raise ValueError(
                f"Expected {len(labels)} rows for {method}, "
                f"found {len(method_rows)}"
            )

        (line,) = axis.plot(
            list(positions),
            method_rows["pF1_mean"],
            marker=marker,
            linewidth=2,
            label=method,
        )

        method_color = line.get_color()

        for position, value in zip(positions, method_rows["pF1_mean"]):
            axis.annotate(
                f"{value:.2f}",
                (position, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor=method_color,
                    edgecolor="none",
                ),
            )

    axis.set_xticks(list(positions), labels)
    axis.set_xlabel("Máximo de afiliações por autor")
    axis.set_ylabel("pF1 (%)")
    axis.set_title(f"{dataset}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    y_min, y_max = axis.get_ylim()
    axis.set_ylim(y_min, y_max + (y_max - y_min) * 0.12)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def load_existing_experiment(
    experiment_dir: Path,
    dataset: str,
) -> bool:
    summary_path = experiment_dir / "summary.csv"

    if not summary_path.exists():
        return False

    summary = pd.read_csv(summary_path)

    required_columns = {
        "method",
        "max_org_affiliation",
        "pF1_mean",
    }
    missing_columns = required_columns - set(summary.columns)

    if missing_columns:
        raise ValueError(
            f"Existing summary.csv is missing columns: {sorted(missing_columns)}"
        )

    expected_methods = {"AND-GloRe", "MCCG"}
    actual_methods = set(summary["method"].dropna().unique())

    if actual_methods != expected_methods:
        raise ValueError(
            f"Existing summary.csv has unexpected methods: " f"{sorted(actual_methods)}"
        )

    expected_values = {
        "unlimited" if value is None else str(value) for value in SWEEP_VALUES
    }
    actual_values = set(summary["max_org_affiliation"].dropna().astype(str).unique())

    if actual_values != expected_values:
        raise ValueError(
            "Existing summary.csv does not contain the expected "
            f"max_org_affiliation values. Expected {sorted(expected_values)}, "
            f"found {sorted(actual_values)}"
        )

    plot_path = experiment_dir / "pf1.png"
    plot_results(summary, plot_path, dataset)

    print(f"Experiment already exists: {experiment_dir}")
    print(f"Loaded summary: {summary_path}")
    print(f"pF1 plot: {plot_path}")

    return True


def main() -> None:
    args = parse_args()

    andglore_config_path = args.andglore_config or Path(
        f"configs/andglore_{args.dataset}.yaml"
    )
    mccg_config_path = args.mccg_config or Path(f"configs/mccg_{args.dataset}.yaml")

    andglore_config = load_yaml_config(str(andglore_config_path))
    mccg_config = load_yaml_config(str(mccg_config_path))

    for path, config in (
        (andglore_config_path, andglore_config),
        (mccg_config_path, mccg_config),
    ):
        if config.get("dataset_name") != args.dataset:
            raise ValueError(f"{path} is for dataset {config.get('dataset_name')!r}")

    experiment_dir = checked_experiment_dir(
        args.experiments_root,
        args.experiment_id,
    )

    if load_existing_experiment(experiment_dir, args.dataset):
        return

    selected_names = args.selected_names

    if selected_names is None:
        selected_names = andglore_config.get("selected_names")

        if selected_names != mccg_config.get("selected_names"):
            raise ValueError(
                "Config selected_names differ; pass --selected-names explicitly"
            )

    manifest = {
        "experiment_id": args.experiment_id,
        "dataset": args.dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_names": selected_names,
        "max_org_affiliation": list(SWEEP_VALUES),
        "configs": {
            "andglore": str(andglore_config_path),
            "mccg": str(mccg_config_path),
        },
        "status": "running",
    }

    manifest_path = experiment_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    (experiment_dir / "configs.yaml").write_text(
        yaml.safe_dump(
            {
                "andglore": andglore_config,
                "mccg": mccg_config,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    preprocessed_path = Path(
        mccg_config.get(
            "preprocessed_path",
            f"data/{args.dataset}/preprocessed.csv",
        )
    )

    features_path = Path(
        mccg_config.get(
            "features_path",
            f"data/{args.dataset}/features.pt",
        )
    )

    preprocessed = load_dataset(str(preprocessed_path))
    embeddings = load_embeddings(str(features_path))

    rows: list[dict[str, Any]] = []

    try:
        for value in SWEEP_VALUES:
            value_label = "unlimited" if value is None else str(value)
            setting_dir = experiment_dir / f"max_org_affiliation_{value_label}"

            andglore_dir = setting_dir / "andglore"
            andglore_dir.mkdir(parents=True)

            andglore_network_path = andglore_dir / "networks.pt"

            networks = build_ambiguous_networks(
                dataset_name=args.dataset,
                preprocessed=preprocessed,
                embeddings=embeddings,
                networks_path=andglore_network_path,
                logs_file=andglore_dir / "network_generation.log",  # type: ignore
                splits=andglore_config.get("splits"),
                selected_names=selected_names,
                max_orgs_per_author=andglore_config.get("max_orgs_per_author"),
                max_author_paper_ratio=andglore_config.get("max_author_paper_ratio"),
                ignore_authors=andglore_config.get("ignore_authors"),
                max_org_affiliation=value,
            )

            andglore_results = andglore_dir / "results.csv"

            run_andglore_experiment(
                dataset_name=args.dataset,
                networks=networks,
                training_args=training_args(andglore_config),
                selected_names=selected_names,
                results_csv_path=str(andglore_results),
                outputs_path=str(andglore_dir / "outputs.pt"),
                log_file_path=str(andglore_dir / "experiment.log"),
            )

            mean = average_pf1(andglore_results)

            rows.append(
                {
                    "method": "AND-GloRe",
                    "max_org_affiliation": value_label,
                    "pF1_mean": mean,
                }
            )

            mccg_dir = setting_dir / "mccg"
            mccg_dir.mkdir(parents=True)

            mccg_network_path = mccg_dir / "networks.pt"

            build_homogeneous_networks_mccg(
                preprocessed_path=preprocessed_path,
                features_path=features_path,
                dataset_name=args.dataset,
                networks_path=mccg_network_path,
                logs_file=mccg_dir / "network_generation.log",
                splits=mccg_config.get("splits"),
                selected_names=selected_names,
                max_orgs_per_author=mccg_config.get("max_orgs_per_author"),
                max_author_paper_ratio=mccg_config.get("max_author_paper_ratio"),
                max_org_affiliation=value,
            )

            mccg_results = mccg_dir / "results.csv"

            run_experiment_mccg(
                dataset_name=args.dataset,
                networks_path=str(mccg_network_path),
                training_args=training_args_mccg(mccg_config),
                selected_names=selected_names,
                results_csv_path=str(mccg_results),
                outputs_path=str(mccg_dir / "outputs.pt"),
                log_file_path=str(mccg_dir / "experiment.log"),
            )

            mean = average_pf1(mccg_results)

            rows.append(
                {
                    "method": "MCCG",
                    "max_org_affiliation": value_label,
                    "pF1_mean": mean,
                }
            )

            pd.DataFrame(rows).to_csv(
                experiment_dir / "summary.csv",
                index=False,
            )

        summary = pd.DataFrame(rows)

        plot_results(
            summary,
            experiment_dir / "pf1.png",
            args.dataset,
        )

        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()

    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise

    finally:
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"Experiment complete: {experiment_dir}")
    print(f"pF1 plot: {experiment_dir / 'pf1.png'}")


if __name__ == "__main__":
    main()
