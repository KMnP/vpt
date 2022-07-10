#!/usr/bin/env python3
"""
evaluate precision@1, @5 equal to Top1 and Top5 error rate
"""
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score
)


def get_continuous_ids(probe_labels: List[int]) -> Dict[int, int]:
    sorted(probe_labels)
    id2continuousid = {}
    for idx, p_id in enumerate(probe_labels):
        id2continuousid[p_id] = idx
    return id2continuousid


def multihot(x: List[List[int]], nb_classes: int) -> np.ndarray:
    """transform to multihot encoding

    Arguments:
        x: list of multi-class integer labels, in the range
            [0, nb_classes-1]
        nb_classes: number of classes for the multi-hot vector

    Returns:
        multihot: multihot vector of type int, (num_samples, nb_classes)
    """
    num_samples = len(x)

    multihot = np.zeros((num_samples, nb_classes), dtype=np.int32)
    for idx, labs in enumerate(x):
        for lab in labs:
            multihot[idx, lab] = 1

    return multihot.astype(np.int)


def compute_map(
        scores: np.ndarray, multihot_targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the mean average precision across all class labels.

    Arguments:
        scores: matrix of per-class distances,
            of size num_samples x nb_classes
        multihot_targets: matrix of multi-hot target predictions,
            of size num_samples x nb_classes

    Returns:
        ap: list of average-precision scores, one for each of
            the nb_classes classes.
        ar: list of average-recall scores, one for each of
            the nb_classes classes.
        mAP: the mean average precision score over all average
            precisions for all nb_classes classes.
        mAR: the mean average recall score over all average
            precisions for all nb_classes classes.
    """
    nb_classes = scores.shape[1]

    ap = np.zeros((nb_classes,), dtype=np.float32)
    ar = np.zeros((nb_classes,), dtype=np.float32)

    for c in range(nb_classes):
        y_true = multihot_targets[:, c]
        y_scores = scores[:, c]

        # Use interpolated average precision (a la PASCAL
        try:
            ap[c] = average_precision_score(y_true, y_scores)
        except ValueError:
            ap[c] = -1

        # Also get the average of the recalls on the raw PR-curve
        try:
            _, rec, _ = precision_recall_curve(y_true, y_scores)
            ar[c] = rec.mean()
        except ValueError:
            ar[c] = -1

    mAP = ap.mean()
    mAR = ar.mean()

    return ap, ar, mAP, mAR


def compute_f1(
        multihot_targets: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> Tuple[float, float, float]:
    # change scores to predict_labels
    predict_labels = scores > threshold
    predict_labels = predict_labels.astype(np.int)

    # change targets to multihot
    f1 = {}
    f1["micro"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="micro"
    )
    f1["samples"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="samples"
    )
    f1["macro"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average="macro"
    )
    f1["none"] = f1_score(
        y_true=multihot_targets,
        y_pred=predict_labels,
        average=None
    )
    return f1["micro"], f1["samples"], f1["macro"], f1["none"]


def get_best_f1_scores(
    multihot_targets: np.ndarray, scores: np.ndarray, threshold_end: int
) -> Dict[str, float]:
    end = 0.5
    end = 0.05
    end = threshold_end
    thrs = np.linspace(
        end, 0.95, int(np.round((0.95 - end) / 0.05)) + 1, endpoint=True
    )
    f1_micros = []
    f1_macros = []
    f1_samples = []
    f1_none = []
    for thr in thrs:
        _micros, _samples, _macros, _none = compute_f1(multihot_targets, scores, thr)
        f1_micros.append(_micros)
        f1_samples.append(_samples)
        f1_macros.append(_macros)
        f1_none.append(_none)

    f1_macros_m = max(f1_macros)
    b_thr = np.argmax(f1_macros)

    f1_micros_m = f1_micros[b_thr]
    f1_samples_m = f1_samples[b_thr]
    f1_none_m = f1_none[b_thr]
    f1 = {}
    f1["micro"] = f1_micros_m
    f1["macro"] = f1_macros_m
    f1["samples"] = f1_samples_m
    f1["threshold"] = thrs[b_thr]
    f1["none"] = f1_none_m
    return f1
