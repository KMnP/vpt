#!/usr/bin/env python3

"""Functions for computing metrics. all metrics has range of 0-1"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, roc_auc_score
)


def accuracy(y_probs, y_true):
    # y_prob: (num_images, num_classes)
    y_preds = np.argmax(y_probs, axis=1)
    accuracy = accuracy_score(y_true, y_preds)
    error = 1.0 - accuracy
    return accuracy, error


def top_n_accuracy(y_probs, truths, n=1):
    # y_prob: (num_images, num_classes)
    # truth: (num_images, num_classes) multi/one-hot encoding
    best_n = np.argsort(y_probs, axis=1)[:, -n:]
    if isinstance(truths, np.ndarray) and truths.shape == y_probs.shape:
        ts = np.argmax(truths, axis=1)
    else:
        # a list of GT class idx
        ts = truths

    num_input = y_probs.shape[0]
    successes = 0
    for i in range(num_input):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / num_input


def compute_acc_auc(y_probs, y_true_ids):
    onehot_tgts = np.zeros_like(y_probs)
    for idx, t in enumerate(y_true_ids):
        onehot_tgts[idx, t] = 1.

    num_classes = y_probs.shape[1]
    if num_classes == 2:
        top1, _ = accuracy(y_probs, y_true_ids)
        # so precision can set all to 2
        try:
            auc = roc_auc_score(onehot_tgts, y_probs, average='macro')
        except ValueError as e:
            print(f"value error encountered {e}, set auc sccore to -1.")
            auc = -1
        return {"top1": top1, "rocauc": auc}

    top1, _ = accuracy(y_probs, y_true_ids)
    k = min([5, num_classes])  # if number of labels < 5, use the total class
    top5 = top_n_accuracy(y_probs, y_true_ids, k)
    return {"top1": top1, f"top{k}": top5}


def topks_correct(preds, labels, ks):
    """Computes the number of top-k correct predictions for each k."""
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [
        top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks
    ]
    return topks_correct


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    if int(labels.min()) < 0:  # has ignore
        keep_ids = np.where(labels.cpu() >= 0)[0]
        preds = preds[keep_ids, :]
        labels = labels[keep_ids]

    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """Computes the top-k accuracy for each k."""
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) for x in num_topks_correct]

