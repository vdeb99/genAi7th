"""Utilities for evaluating router decisions against a labeled query set."""
from collections import Counter
from typing import Iterable


def routing_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict:
    truth, pred = list(y_true), list(y_pred)
    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    labels = ["way1", "way2"]
    confusion = {actual: {guess: 0 for guess in labels} for actual in labels}
    for actual, guess in zip(truth, pred):
        if actual in labels and guess in labels:
            confusion[actual][guess] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": sum(confusion[label].values())}
    accuracy = sum(confusion[label][label] for label in labels) / len(truth) if truth else 0.0
    return {"accuracy": round(accuracy, 4), "confusion_matrix": confusion, "per_class": per_class, "count": len(truth)}
