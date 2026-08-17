import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from andglore.baseline.mccg.augmentation_mccg import (
    adjacency_mccg,
    centralities_mccg,
    diffusion_mccg,
    drop_edges_weighted_mccg,
    drop_features_weighted_mccg,
)
from andglore.baseline.mccg.model_mccg import GATMCCG, MCCG
from andglore.baseline.mccg.training_args_mccg import TrainingArgsMCCG
from andglore.evaluation.metrics import ari_evaluate, bcubed_evaluate, pairwise_evaluate
from andglore.trainer.logger import Logger
from andglore.utils import AVERAGE_NAME, save_csv_results, set_seed


def _resolve_device_mccg(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _edge_index_mccg(
    network: dict[str, Any], training_args: TrainingArgsMCCG
) -> torch.Tensor:
    pair_index = network["pair_index"].long()
    mask = (
        (network["author_overlap"] >= training_args.th_a)
        | (network["org_jaccard"] >= training_args.th_o)
        | (network["venue_overlap"] >= training_args.th_v)
    )
    pair_index = pair_index[:, mask]
    if pair_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat([pair_index, pair_index.flip(0)], dim=1)


def _cluster_mccg(embeddings: torch.Tensor, eps: float, minimum: int) -> np.ndarray:
    count = embeddings.shape[0]
    if count < minimum:
        return np.zeros(count, dtype=np.int64)
    distances = pairwise_distances(
        embeddings.detach().cpu().numpy(), metric="cosine"
    ).astype("double")
    return HDBSCAN(
        cluster_selection_epsilon=eps,
        min_samples=minimum,
        min_cluster_size=minimum,
        metric="precomputed",
    ).fit_predict(distances)


def _prediction_labels_mccg(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64, copy=True)
    if np.any(labels < 0):
        next_cluster = int(labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0
        labels[labels < 0] = next_cluster
    return labels


def _train_name_mccg(
    network: dict[str, Any],
    training_args: TrainingArgsMCCG,
    device: torch.device,
    progress: tqdm,
    name_position: str,
    run_position: str,
    seed: int,
    live_pf1: float,
) -> tuple[dict[str, float], dict[str, Any], float]:
    features = network["features"].float().to(device)
    paper_labels = network["labels"].long().cpu()
    num_nodes, feature_dim = features.shape
    if feature_dim != training_args.layer_shape[0]:
        raise ValueError(
            f"MCCG layer_shape input is {training_args.layer_shape[0]}, but "
            f"{network['name']!r} has {feature_dim}-dimensional features"
        )

    edge_index = _edge_index_mccg(network, training_args).to(device)
    full_adjacency = adjacency_mccg(edge_index, num_nodes)
    full_diffusion = diffusion_mccg(full_adjacency, steps=training_args.diffusion_steps)

    edge_weights, feature_weights = centralities_mccg(
        edge_index, features, num_nodes, training_args.drop_scheme
    )
    edge_index_view1 = drop_edges_weighted_mccg(
        edge_index,
        edge_weights,
        training_args.drop_edge_rate_view1,
        training_args.augmentation_threshold,
    )
    edge_index_view2 = drop_edges_weighted_mccg(
        edge_index,
        edge_weights,
        training_args.drop_edge_rate_view2,
        training_args.augmentation_threshold,
    )
    adjacency_view1 = adjacency_mccg(edge_index_view1, num_nodes)
    adjacency_view2 = adjacency_mccg(edge_index_view2, num_nodes)
    diffusion_view1 = diffusion_mccg(
        adjacency_view1, steps=training_args.diffusion_steps
    )
    diffusion_view2 = diffusion_mccg(
        adjacency_view2, steps=training_args.diffusion_steps
    )
    features_view1 = drop_features_weighted_mccg(
        features,
        feature_weights,
        training_args.drop_feature_rate_view1,
        training_args.augmentation_threshold,
    )
    features_view2 = drop_features_weighted_mccg(
        features,
        feature_weights,
        training_args.drop_feature_rate_view2,
        training_args.augmentation_threshold,
    )

    encoder_mccg = GATMCCG(
        input_dim=training_args.layer_shape[0],
        hidden_dim=training_args.layer_shape[1],
        output_dim=training_args.layer_shape[2],
        alpha=training_args.gat_alpha,
        dropout=training_args.gat_dropout,
    )
    model_mccg = MCCG(
        encoder_mccg=encoder_mccg,
        hidden_dim=training_args.layer_shape[2],
        multiview_projection_dim=training_args.dim_proj_multiview,
        cluster_projection_dim=training_args.dim_proj_cluster,
    ).to(device)
    optimizer_mccg = torch.optim.Adam(  # type: ignore
        model_mccg.parameters(),
        lr=training_args.lr,
        weight_decay=training_args.l2_coef,
    )

    started = time.time()
    final_loss = float("nan")
    for epoch in range(1, training_args.epochs + 1):
        model_mccg.train()
        optimizer_mccg.zero_grad()
        multiview_embedding, cluster_embedding = model_mccg(
            features_view1,
            adjacency_view1,
            diffusion_view1,
            features_view2,
            adjacency_view2,
            diffusion_view2,
        )
        pseudo_labels = torch.from_numpy(
            _cluster_mccg(cluster_embedding, training_args.db_eps, training_args.db_min)
        ).to(device)
        cluster_loss = model_mccg.self_supervised_contrastive_loss_mccg(
            cluster_embedding.unsqueeze(1),
            pseudo_labels,
            contrast_mode="one",
            temperature=training_args.t_cluster,
        )
        multiview_loss = model_mccg.self_supervised_contrastive_loss_mccg(
            multiview_embedding,
            pseudo_labels,
            contrast_mode="all",
            temperature=training_args.t_multiview,
        )
        loss = (
            training_args.w_cluster * cluster_loss
            + (1 - training_args.w_cluster) * multiview_loss
        )
        loss.backward()
        optimizer_mccg.step()
        final_loss = loss.item()
        progress.set_postfix(
            name=name_position,
            epoch=f"{epoch}/{training_args.epochs}",
            loss=f"{loss.item():.4f}",
            run=run_position,
            seed=seed,
            avg_pf1=f"{live_pf1:.2%}",
        )

    runtime = int(time.time() - started)
    with torch.no_grad():
        model_mccg.eval()
        embeddings = model_mccg.encoder_mccg(features, full_adjacency, full_diffusion)
        embeddings = F.normalize(model_mccg.cluster_projector_mccg(embeddings), dim=1)
        predictions = _prediction_labels_mccg(
            _cluster_mccg(embeddings, training_args.db_eps, training_args.db_min)
        )

    pairwise = pairwise_evaluate(paper_labels, predictions)
    bcubed = bcubed_evaluate(paper_labels, predictions)
    ari = ari_evaluate(paper_labels, predictions)
    metrics = {
        "pP": pairwise[0],
        "pR": pairwise[1],
        "pF1": pairwise[2],
        "bP": bcubed[0],
        "bR": bcubed[1],
        "bF1": bcubed[2],
        "ari": ari,
        "runtime": runtime,
    }
    output = {
        "output": embeddings.detach().cpu(),
        "pred": predictions,
        "labels": paper_labels,
        "paper_ids": network["paper_ids"],
    }
    return metrics, output, final_loss


def run_experiment_mccg(
    dataset_name: str,
    networks_path: str,
    training_args: TrainingArgsMCCG,
    log_file_path: Optional[str] = None,
    results_csv_path: Optional[str] = None,
    outputs_path: Optional[str] = None,
    selected_names: Optional[list[str]] = None,
) -> None:
    for path in (log_file_path, results_csv_path, outputs_path):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    logger = Logger(log_file_path).logger
    logger.info(
        "Starting MCCG Experiment", extra={"show_time": True, "break_line": True}
    )
    logger.info(f"Experiment Args:\n{training_args}", extra={"break_line": True})

    networks = torch.load(networks_path, map_location="cpu", weights_only=False)
    networks_by_name = {network["name"]: network for network in networks}
    names = selected_names if selected_names is not None else sorted(networks_by_name)
    missing_names = sorted(set(names).difference(networks_by_name))
    if missing_names:
        raise ValueError(f"Selected names not found in MCCG networks: {missing_names}")
    if not names:
        raise ValueError("No MCCG name networks were selected")

    device = _resolve_device_mccg(training_args.device)
    if str(device) != training_args.device:
        logger.warning(
            f"Requested device {training_args.device!r} is unavailable; using CPU"
        )
    results: dict[int, dict[str, dict[str, float]]] = {}
    outputs: dict[int, dict[str, dict[str, Any]]] = {}
    progress = tqdm(training_args.seeds, desc="MCCG Experiment")

    for run, seed in enumerate(progress, start=1):
        results[seed] = {}
        outputs[seed] = {}
        live_pf1 = 0.0
        for name_index, name in enumerate(names, start=1):
            set_seed(seed)
            completed_names = name_index - 1
            current_avg_pf1 = live_pf1 / completed_names if completed_names > 0 else 0.0
            logger.info(
                f"Run {run}/{len(training_args.seeds)} - "
                f"Name {name_index}/{len(names)}: {name} - "
                f"Avg. F1: {current_avg_pf1:.4f}",
                extra={"show_time": True},
            )
            metrics, output, final_loss = _train_name_mccg(
                networks_by_name[name],
                training_args,
                device,
                progress,
                f"{name_index}/{len(names)}:{name}",
                f"{run}/{len(training_args.seeds)}",
                seed,
                current_avg_pf1,
            )
            results[seed][name] = metrics
            outputs[seed][name] = output
            live_pf1 += metrics["pF1"]
            progress.set_postfix(
                name=f"{name_index}/{len(names)}:{name}",
                epoch=f"{training_args.epochs}/{training_args.epochs}",
                loss=f"{final_loss:.4f}",
                run=f"{run}/{len(training_args.seeds)}",
                seed=seed,
                avg_pf1=f"{(live_pf1 / name_index):.2%}",
            )
            logger.info(f"Results for name: {name}", extra={"show_time": True})
            logger.info(f"Runtime: {timedelta(seconds=int(metrics['runtime']))}")
            logger.info(
                f"pP: {metrics['pP']:.2%} pR: {metrics['pR']:.2%} "
                f"pF1: {metrics['pF1']:.2%}"
            )
            logger.info(
                f"bP: {metrics['bP']:.2%} bR: {metrics['bR']:.2%} "
                f"bF1: {metrics['bF1']:.2%}"
            )
            logger.info(f"ARI: {metrics['ari']:.2%}\n")

    results_frame = save_csv_results(results, results_csv_path, outputs, outputs_path)
    overall = results_frame[results_frame["name"] == AVERAGE_NAME].iloc[0]
    logger.info(
        "Average MCCG Results Across All Names",
        extra={"show_time": True, "print": True},
    )
    logger.info(
        f"Runtime: {timedelta(seconds=int(overall['runtime_mean']))} "
        f"(± {timedelta(seconds=int(overall['runtime_std']))})",
        extra={"print": True},
    )
    logger.info(
        f"pP: {overall['pP_mean']:.2%} (± {overall['pP_std']:.2%}) "
        f"pR: {overall['pR_mean']:.2%} (± {overall['pR_std']:.2%}) "
        f"pF1: {overall['pF1_mean']:.2%} (± {overall['pF1_std']:.2%})",
        extra={"print": True},
    )
    logger.info(
        f"bP: {overall['bP_mean']:.2%} (± {overall['bP_std']:.2%}) "
        f"bR: {overall['bR_mean']:.2%} (± {overall['bR_std']:.2%}) "
        f"bF1: {overall['bF1_mean']:.2%} (± {overall['bF1_std']:.2%})",
        extra={"print": True},
    )
    logger.info(
        f"ARI: {overall['ari_mean']:.2%} (± {overall['ari_std']:.2%})",
        extra={"print": True},
    )
    logger.info("", extra={"break_line": True, "print": True})
