from typing import Union

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score


def bcubed_evaluate(
    labels: Union[np.ndarray, list, torch.Tensor],
    pred_labels: Union[np.ndarray, list, torch.Tensor],
):
    """Calculate B-Cubed precision, recall, and F1 for clustering results."""

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()

    labels = np.asarray(labels)
    pred_labels = np.asarray(pred_labels)
    n = len(labels)

    if n == 0:
        return 0.0, 0.0, 0.0

    precision_sum = 0.0
    recall_sum = 0.0

    for i in range(n):
        same_label = labels == labels[i]
        same_pred = pred_labels == pred_labels[i]

        correctly_related = np.logical_and(same_label, same_pred).sum()

        precision_sum += correctly_related / same_pred.sum()
        recall_sum += correctly_related / same_label.sum()

    precision = precision_sum / n
    recall = recall_sum / n
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def ari_evaluate(
    labels: Union[np.ndarray, list, torch.Tensor],
    pred_labels: Union[np.ndarray, list, torch.Tensor],
):
    """Calculate Adjusted Rand Index (ARI) for clustering results."""

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()

    labels = np.asarray(labels)
    pred_labels = np.asarray(pred_labels)
    return adjusted_rand_score(labels, pred_labels)


def pairwise_evaluate(
    labels: Union[np.ndarray, list, torch.Tensor],
    pred_labels: Union[np.ndarray, list, torch.Tensor],
):
    """Calculate pairwise precision, recall, and F1 for clustering results."""

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()

    labels = np.asarray(labels)
    pred_labels = np.asarray(pred_labels)
    if len(labels) != len(pred_labels):
        return 0.0, 0.0, 0.0

    tp, fp, fn = 0.0, 0.0, 0.0
    n = len(labels)

    for i in range(n):
        for j in range(i + 1, n):
            label_match = (labels[i] == labels[j]) and (labels[i] != -1)
            pred_match = (pred_labels[i] == pred_labels[j]) and (pred_labels[i] != -1)
            if label_match and pred_match:
                tp += 1
            elif not label_match and pred_match:
                fp += 1
            elif label_match and not pred_match:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1
