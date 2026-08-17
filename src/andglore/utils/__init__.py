import argparse
import random
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml

AVERAGE_NAME = "- AVERAGE -"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Invalid config format: top-level YAML value must be a map")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AND-GloRe experiment from a YAML config file."
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--max_orgs_per_author",
        type=int,
        default=None,
        help="Maximum number of organizations allowed per author.",
    )

    parser.add_argument(
        "--max_author_paper_ratio",
        type=float,
        default=None,
        help="Maximum author-to-paper ratio.",
    )

    args = parser.parse_args()

    if not args.config:
        parser.print_help()
        parser.exit(
            2,
            "\nError: no config provided. Use --config <path-to-yml>.\n",
        )

    return args


def save_csv_results(
    results: dict,
    results_csv_path: Optional[str],
    outputs: dict,
    output_file: Optional[str],
) -> pd.DataFrame:

    if output_file is not None:
        torch.save(outputs, output_file)

    rows = []
    for seed, name_results in results.items():
        for name, metrics in name_results.items():
            rows.append({"seed": seed, "name": name, **metrics})

    if not rows:
        raise ValueError("`results` is empty, nothing to aggregate.")

    results_df = pd.DataFrame(rows)

    # Mean and std across seeds, per name
    grouped = results_df.drop(columns="seed").groupby("name")
    summary_df = grouped.agg(["mean", "std"])
    summary_df.columns = [f"{metric}_{stat}" for metric, stat in summary_df.columns]
    summary_df = summary_df.reset_index()

    # Replace NaN std (occurs when only one observation) with 0
    std_cols = [c for c in summary_df.columns if c.endswith("_std")]
    summary_df[std_cols] = summary_df[std_cols].fillna(0.0)

    # Overall row averaged across all names
    metrics_df = results_df.drop(columns=["seed", "name"])
    overall_row = {"name": AVERAGE_NAME}
    for metric in metrics_df.columns:
        overall_row[f"{metric}_mean"] = metrics_df[metric].mean()  # type: ignore
        overall_row[f"{metric}_std"] = metrics_df[metric].std(ddof=1)  # type: ignore
        if pd.isna(overall_row[f"{metric}_std"]):
            overall_row[f"{metric}_std"] = 0.0  # type: ignore

    summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)

    if results_csv_path is not None:
        csv_df = summary_df.copy()
        percentage_columns = [
            column
            for column in csv_df.columns
            if column != "name" and not column.startswith("runtime_")
        ]
        csv_df[percentage_columns] = csv_df[percentage_columns] * 100
        csv_df.to_csv(results_csv_path, index=False, float_format="%.2f")

    return summary_df
