import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances

from andglore.evaluation.metrics import ari_evaluate, bcubed_evaluate, pairwise_evaluate


def hdbscan_evaluation(
    embeddings,
    paper_labels,
    cluster_selection_epsilon: float,
    min_cluster_size: int,
):
    """Cluster AND-GloRe embeddings using the same HDBSCAN setup as MCCG."""
    embeddings = np.asarray(
        embeddings.detach().cpu() if hasattr(embeddings, "detach") else embeddings
    )
    n_samples = embeddings.shape[0]

    if n_samples < min_cluster_size:
        predicted_labels = np.zeros(n_samples, dtype=np.int64)
    else:
        distances = pairwise_distances(embeddings, metric="cosine").astype("double")
        predicted_labels = HDBSCAN(
            cluster_selection_epsilon=cluster_selection_epsilon,
            min_samples=min_cluster_size,
            min_cluster_size=min_cluster_size,
            metric="precomputed",
        ).fit_predict(distances)

        # Match MCCG's evaluation behavior: treat all HDBSCAN noise points as
        # one additional cluster instead of exposing label -1 to the metrics.
        if np.any(predicted_labels < 0):
            next_cluster = (
                int(predicted_labels[predicted_labels >= 0].max() + 1)
                if np.any(predicted_labels >= 0)
                else 0
            )
            predicted_labels[predicted_labels < 0] = next_cluster

    pairwise = pairwise_evaluate(paper_labels, predicted_labels)
    bcubed = bcubed_evaluate(paper_labels, predicted_labels)
    ari = ari_evaluate(paper_labels, predicted_labels)

    return (
        predicted_labels,
        float("nan"),
        cluster_selection_epsilon,
        pairwise,
        bcubed,
        ari,
    )


def adaptative_hac_evaluation(
    embeddings,
    paper_labels,
    min_distance_threshold: float,
    max_distance_threshold: float,
    step: float | None = None,
):
    if step is None:
        step = (max_distance_threshold - min_distance_threshold) / 100

    pred = np.array([-1] * embeddings.shape[0])
    best_score = float("-inf")
    best_pairwise = (-1.0, -1.0, -1.0)
    best_bcubed = (-1.0, -1.0, -1.0)
    best_ari = -1.0
    best_threshold = min_distance_threshold

    thresholds = np.arange(
        min_distance_threshold,
        max_distance_threshold + step / 2,
        step,
    )

    n_samples = embeddings.shape[0]

    fallback_set = False

    for threshold in thresholds:
        predicted_labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        ).fit_predict(embeddings)

        pairwise = pairwise_evaluate(
            paper_labels,
            predicted_labels,
        )

        bcubed = bcubed_evaluate(
            paper_labels,
            predicted_labels,
        )

        ari = ari_evaluate(
            paper_labels,
            predicted_labels,
        )

        n_clusters = len(np.unique(predicted_labels))

        if not fallback_set:
            pred = predicted_labels
            best_pairwise = pairwise
            best_bcubed = bcubed
            best_ari = ari
            best_threshold = threshold
            fallback_set = True

        if 2 <= n_clusters < n_samples:
            score = silhouette_score(
                embeddings,
                predicted_labels,
                metric="cosine",
            )

            if score > best_score:
                best_score = score
                best_pairwise = pairwise
                best_bcubed = bcubed
                best_ari = ari
                best_threshold = threshold
                pred = predicted_labels

    return (
        pred,
        best_score,
        best_threshold,
        best_pairwise,
        best_bcubed,
        best_ari,
    )
