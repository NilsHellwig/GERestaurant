import datetime
import json
import os

import pandas as pd


def format_seconds_to_time_string(total_seconds: float) -> str:
    """Format total seconds into a human-readable duration string (e.g., '1h 2m 3s')."""
    time_duration = datetime.timedelta(seconds=total_seconds)
    hours, remainder = divmod(int(time_duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    time_format = []
    if hours > 0:
        time_format.append(f"{hours}h")
    if minutes > 0:
        time_format.append(f"{minutes}m")
    time_format.append(f"{seconds}s")

    return " ".join(time_format)


def divide(a: float, b: float) -> float:
    """Divide a by b, returning 0 if b is 0."""
    return a / b if b > 0 else 0


def compute_popular_metrics(tp: int, tn: int, fp: int, fn: int) -> tuple:
    """Compute precision, recall, and F1-score from confusion matrix components."""
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    f1 = divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def save_pred_and_labels(predictions, labels, results: dict):
    """Save model predictions and ground truth labels to a JSON file."""
    os.makedirs("split_results", exist_ok=True)
    file_name = f"evaluation_{results.get('TARGET', 'unknown')}.json"
    file_path = os.path.join("split_results", file_name)

    with open(file_path, "w") as f:
        json.dump(
            {"predictions": predictions.tolist(), "labels": labels.tolist()},
            f,
            indent=4,
        )

