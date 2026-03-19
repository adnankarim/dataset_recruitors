import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = str(BASE_DIR / "output" / "feature_splits")
DEFAULT_STORE_DIR = str(BASE_DIR / "output" / "store")
DEFAULT_RUN_DIR = str(BASE_DIR / "training_runs" / "dense_embedding_classifier")
DEFAULT_METADATA_COLUMNS = "event_id,user_id,search_id,profil_id,split"
DEFAULT_LABEL_ORDER = "Go,Interesting,Why not,Not interesting,Out of scope"
DEFAULT_MERGED_LABEL_ORDER = "Go,Interesting / Why not,Not interesting / Out of scope"
DEFAULT_BINARY_LABEL_ORDER = "Go / Interesting,Why not / Not interesting / Out of scope"
MIN_SAMPLE_WEIGHT = 1e-6
INT8_SCALE = 127.0
EMBEDDING_ID_COLUMNS = ("query_text_id", "user_text_id", "candidate_text_id")
MERGED_THREE_LABEL_MAP = {
    "Go": "Go",
    "Interesting": "Interesting / Why not",
    "Why not": "Interesting / Why not",
    "Not interesting": "Not interesting / Out of scope",
    "Out of scope": "Not interesting / Out of scope",
}
MERGED_TWO_LABEL_MAP = {
    "Go": "Go / Interesting",
    "Interesting": "Go / Interesting",
    "Why not": "Why not / Not interesting / Out of scope",
    "Not interesting": "Why not / Not interesting / Out of scope",
    "Out of scope": "Why not / Not interesting / Out of scope",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative float.")
    return parsed


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [positive_int(item) for item in parse_csv_list(value)]


def safe_round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def safe_metric(fn, *args, **kwargs) -> float | None:
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return None


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def read_split(feature_dir: Path, split_name: str) -> pd.DataFrame:
    path = feature_dir / f"{split_name}_with_pplx_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature split: {path}")
    return pd.read_parquet(path)


def load_embedding_store(store_dir: Path) -> dict[str, np.ndarray]:
    store = {}
    for role in ("query", "user", "candidate"):
        path = store_dir / f"{role}_embeddings.int8.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding store file: {path}")
        store[role] = np.load(path, mmap_mode="r")
    return store


def resolve_coldstart_mask(df: pd.DataFrame) -> np.ndarray:
    if "embedding_user_available" in df.columns:
        values = pd.to_numeric(df["embedding_user_available"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        return values <= 0
    if "is_coldstart" in df.columns:
        values = df["is_coldstart"]
        if values.dtype == bool:
            return values.to_numpy()
        return values.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
    if "user_profile_text" in df.columns:
        return df["user_profile_text"].astype(str).str.lower().eq("coldstart").to_numpy()
    return np.zeros(len(df), dtype=bool)


def generate_feature_columns(embedding_prefix_dim: int) -> list[str]:
    return (
        [f"query_emb_{index:04d}" for index in range(embedding_prefix_dim)]
        + [f"user_emb_{index:04d}" for index in range(embedding_prefix_dim)]
        + [f"candidate_emb_{index:04d}" for index in range(embedding_prefix_dim)]
    )


def build_raw_embedding_feature_matrix(
    df: pd.DataFrame,
    embedding_store: dict[str, np.ndarray],
    embedding_prefix_dim: int,
) -> tuple[np.ndarray, int]:
    query_ids = df["query_text_id"].to_numpy(dtype=np.int32)
    user_ids = df["user_text_id"].to_numpy(dtype=np.int32)
    candidate_ids = df["candidate_text_id"].to_numpy(dtype=np.int32)

    max_dim = int(embedding_store["query"].shape[1])
    if embedding_prefix_dim <= 0 or embedding_prefix_dim > max_dim:
        raise ValueError(f"embedding_prefix_dim must be in 1..{max_dim}, got {embedding_prefix_dim}")

    q = np.asarray(embedding_store["query"][query_ids, :embedding_prefix_dim], dtype=np.int8)
    u = np.asarray(embedding_store["user"][user_ids, :embedding_prefix_dim], dtype=np.int8)
    c = np.asarray(embedding_store["candidate"][candidate_ids, :embedding_prefix_dim], dtype=np.int8)

    coldstart_mask = resolve_coldstart_mask(df)
    if np.any(coldstart_mask):
        u[coldstart_mask] = 0

    matrix = np.concatenate([q, u, c], axis=1)
    return matrix, int(np.sum(coldstart_mask))


def split_labels_and_metadata(
    df: pd.DataFrame,
    target_column: str,
    weight_column: str | None,
    metadata_columns: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, pd.DataFrame]:
    labels = df[target_column].to_numpy() if target_column in df.columns else None
    weights = df[weight_column].to_numpy(dtype=np.float32) if weight_column and weight_column in df.columns else None
    metadata = df[[column for column in metadata_columns if column in df.columns]].copy()
    return labels, weights, metadata


def merge_labels(y: np.ndarray | None, label_mode: str) -> np.ndarray | None:
    if y is None:
        return None
    series = pd.Series(y).astype(str)
    if label_mode == "original":
        return series.to_numpy(dtype=object)
    if label_mode == "merged3":
        label_map = MERGED_THREE_LABEL_MAP
    elif label_mode == "merged2":
        label_map = MERGED_TWO_LABEL_MAP
    else:
        raise ValueError(f"Unsupported label mode: {label_mode}")
    unknown = sorted(set(series.unique()) - set(label_map))
    if unknown:
        raise ValueError(f"Cannot apply {label_mode} label mode because these labels are unmapped: {unknown}")
    return series.map(label_map).to_numpy(dtype=object)


def infer_label_classes(y: np.ndarray, preferred_order: list[str] | None = None) -> list[str]:
    values = pd.Series(y).dropna().astype(str)
    unique_values = values.unique().tolist()
    if preferred_order:
        preferred_set = set(preferred_order)
        ordered = [label for label in preferred_order if label in unique_values]
        remaining = sorted(label for label in unique_values if label not in preferred_set)
        unique_values = ordered + remaining
    else:
        unique_values = sorted(unique_values)
    if len(unique_values) < 2:
        raise ValueError(f"Expected at least 2 unique labels. Found: {unique_values}")
    return unique_values


def encode_labels(y: np.ndarray | None, class_to_index: dict[str, int]) -> np.ndarray | None:
    if y is None:
        return None
    series = pd.Series(y).astype(str)
    unknown = sorted(set(series.unique()) - set(class_to_index))
    if unknown:
        raise ValueError(f"Found labels that were not present in the training split: {unknown}")
    return series.map(class_to_index).to_numpy(dtype=np.int64)


def decode_label_indices(indices: np.ndarray, class_names: list[str]) -> np.ndarray:
    return np.asarray([class_names[int(index)] for index in indices], dtype=object)


def sanitize_label_name(label: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip().lower()).strip("_")
    return sanitized or "label"


def compute_class_weight_vector(y_encoded: np.ndarray, num_classes: int, mode: str) -> np.ndarray | None:
    if mode == "off":
        return None
    if mode != "balanced-sample-weight":
        raise ValueError(f"Unsupported class imbalance handling mode: {mode}")
    counts = np.bincount(y_encoded, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Cannot compute class-balanced sample weights when a class has zero training examples.")
    total = counts.sum()
    return (total / (num_classes * counts)).astype(np.float32)


def combine_sample_weights(
    base_weights: np.ndarray | None,
    y_encoded: np.ndarray | None,
    class_weight_vector: np.ndarray | None,
) -> np.ndarray | None:
    if y_encoded is None:
        return base_weights
    combined = None
    if class_weight_vector is not None:
        combined = class_weight_vector[y_encoded].astype(np.float32)
    if base_weights is not None:
        combined = base_weights.astype(np.float32) if combined is None else combined * base_weights.astype(np.float32)
    return combined


def sanitize_positive_weights(
    weights: np.ndarray | None,
    min_value: float = MIN_SAMPLE_WEIGHT,
) -> tuple[np.ndarray | None, dict]:
    if weights is None:
        return None, {
            "available": False,
            "count": 0,
            "replaced_non_finite": 0,
            "replaced_non_positive": 0,
            "min_applied_weight": min_value,
        }

    sanitized = np.asarray(weights, dtype=np.float32).copy()
    non_finite_mask = ~np.isfinite(sanitized)
    non_positive_mask = sanitized <= 0
    replace_mask = non_finite_mask | non_positive_mask
    if np.any(replace_mask):
        sanitized[replace_mask] = min_value
    stats = {
        "available": True,
        "count": int(len(sanitized)),
        "replaced_non_finite": int(np.sum(non_finite_mask)),
        "replaced_non_positive": int(np.sum(non_positive_mask & ~non_finite_mask)),
        "min_applied_weight": min_value,
    }
    return sanitized, stats


def multiclass_macro_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None,
    num_classes: int,
) -> float | None:
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    return safe_metric(roc_auc_score, y_bin, y_prob, average="macro", multi_class="ovr", sample_weight=sample_weight)


def multiclass_macro_pr_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None,
    num_classes: int,
) -> float | None:
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))
    return safe_metric(average_precision_score, y_bin, y_prob, average="macro", sample_weight=sample_weight)


def evaluate_split(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None,
    class_names: list[str],
) -> dict[str, float | None]:
    y_pred = np.argmax(y_prob, axis=1).astype(np.int64)
    num_classes = len(class_names)
    metrics = {
        "count": int(len(y_true)),
        "num_classes": int(num_classes),
        "logloss": safe_metric(log_loss, y_true, y_prob, sample_weight=sample_weight, labels=np.arange(num_classes)),
        "accuracy": safe_metric(accuracy_score, y_true, y_pred, sample_weight=sample_weight),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred, sample_weight=sample_weight),
        "precision_macro": safe_metric(
            precision_score, y_true, y_pred, average="macro", sample_weight=sample_weight, zero_division=0
        ),
        "recall_macro": safe_metric(
            recall_score, y_true, y_pred, average="macro", sample_weight=sample_weight, zero_division=0
        ),
        "f1_macro": safe_metric(f1_score, y_true, y_pred, average="macro", sample_weight=sample_weight, zero_division=0),
        "f1_weighted": safe_metric(
            f1_score, y_true, y_pred, average="weighted", sample_weight=sample_weight, zero_division=0
        ),
        "roc_auc_ovr_macro": multiclass_macro_roc_auc(y_true, y_prob, sample_weight, num_classes),
        "pr_auc_ovr_macro": multiclass_macro_pr_auc(y_true, y_prob, sample_weight, num_classes),
    }
    return {key: safe_round(value) if isinstance(value, float) else value for key, value in metrics.items()}


def save_prediction_frame(
    output_path: Path,
    metadata: pd.DataFrame,
    y_true: np.ndarray | None,
    y_true_labels: np.ndarray | None,
    original_y_true_labels: np.ndarray | None,
    sample_weight: np.ndarray | None,
    effective_sample_weight: np.ndarray | None,
    y_prob: np.ndarray,
    class_names: list[str],
) -> None:
    frame = metadata.copy()
    if y_true is not None:
        frame["label_id"] = y_true
    if y_true_labels is not None:
        frame["label"] = y_true_labels
    if original_y_true_labels is not None:
        frame["original_label"] = original_y_true_labels
    if sample_weight is not None:
        frame["label_weight"] = sample_weight
    if effective_sample_weight is not None:
        frame["effective_sample_weight"] = effective_sample_weight
    predicted_indices = np.argmax(y_prob, axis=1).astype(np.int64)
    frame["predicted_label_id"] = predicted_indices
    frame["predicted_label"] = decode_label_indices(predicted_indices, class_names)
    frame["prediction_confidence"] = y_prob.max(axis=1)
    for class_index, class_name in enumerate(class_names):
        frame[f"prob_{sanitize_label_name(class_name)}"] = y_prob[:, class_index]
    frame.to_parquet(output_path, index=False)


def save_metrics_table(metrics_by_split: dict[str, dict], output_path: Path) -> None:
    frame = pd.DataFrame.from_dict(metrics_by_split, orient="index")
    frame.index.name = "split"
    frame.reset_index().to_csv(output_path, index=False)


def plot_label_distribution(labels_by_split: dict[str, np.ndarray | None], output_path: Path) -> None:
    available = {split: labels for split, labels in labels_by_split.items() if labels is not None}
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for axis, (split_name, labels) in zip(axes, available.items()):
        counts = pd.Series(labels).value_counts().sort_index()
        colors = plt.cm.tab10(np.linspace(0, 1, len(counts)))
        axis.bar([str(index) for index in counts.index], counts.values, color=colors)
        axis.set_title(f"{split_name} label distribution")
        axis.set_xlabel("label")
        axis.set_ylabel("count")
        axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(history: dict[str, list[float]], output_path: Path) -> None:
    if not history.get("epoch"):
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    epochs = history["epoch"]

    axes[0].plot(epochs, history.get("train_loss", []), label="train_loss")
    axes[0].plot(epochs, history.get("valid_loss", []), label="valid_loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history.get("valid_accuracy", []), label="valid_accuracy")
    axes[1].plot(epochs, history.get("valid_balanced_accuracy", []), label="valid_balanced_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, history.get("valid_f1_macro", []), label="valid_f1_macro")
    axes[2].plot(epochs, history.get("valid_precision_macro", []), label="valid_precision_macro")
    axes[2].plot(epochs, history.get("valid_recall_macro", []), label="valid_recall_macro")
    axes[2].set_title("Macro Metrics")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    axes[3].plot(epochs, history.get("valid_logloss", []), label="valid_logloss")
    axes[3].plot(epochs, history.get("valid_roc_auc_ovr_macro", []), label="valid_roc_auc_ovr_macro")
    axes[3].plot(epochs, history.get("valid_pr_auc_ovr_macro", []), label="valid_pr_auc_ovr_macro")
    axes[3].set_title("Validation Probability Metrics")
    axes[3].grid(alpha=0.3)
    axes[3].legend()

    axes[2].set_xlabel("epoch")
    axes[3].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(predictions: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path, num_classes: int) -> None:
    if not predictions:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False
    for split_name, (y_true, y_prob) in predictions.items():
        if len(np.unique(y_true)) < 2:
            continue
        y_bin = label_binarize(y_true, classes=np.arange(num_classes))
        try:
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
            auc_value = roc_auc_score(y_bin, y_prob, average="micro", multi_class="ovr")
        except ValueError:
            continue
        ax.plot(fpr, tpr, label=f"{split_name} micro-OVR (AUC={auc_value:.4f})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(predictions: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path, num_classes: int) -> None:
    if not predictions:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False
    for split_name, (y_true, y_prob) in predictions.items():
        y_bin = label_binarize(y_true, classes=np.arange(num_classes))
        try:
            precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
            ap_value = average_precision_score(y_bin, y_prob, average="micro")
        except ValueError:
            continue
        ax.plot(recall, precision, label=f"{split_name} micro-OVR (AP={ap_value:.4f})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prediction_histograms(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    if not predictions:
        return
    fig, axes = plt.subplots(len(predictions), 1, figsize=(9, max(4.5, 4 * len(predictions))), sharex=True)
    if len(predictions) == 1:
        axes = [axes]
    for axis, (split_name, (y_true, y_prob)) in zip(axes, predictions.items()):
        predicted = np.argmax(y_prob, axis=1)
        confidence = y_prob.max(axis=1)
        correct = predicted == y_true
        axis.hist(confidence[correct], bins=40, alpha=0.6, label="correct", color="#1f77b4")
        axis.hist(confidence[~correct], bins=40, alpha=0.6, label="incorrect", color="#d62728")
        axis.set_title(f"Prediction Confidence Distribution: {split_name}")
        axis.set_ylabel("count")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Predicted confidence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration_curves(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    if not predictions:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for split_name, (y_true, y_prob) in predictions.items():
        predicted = np.argmax(y_prob, axis=1)
        confidence = y_prob.max(axis=1)
        correctness = (predicted == y_true).astype(np.int8)
        quantiles = np.linspace(0, 1, 11)
        bins = np.quantile(confidence, quantiles)
        bins[0] = 0.0
        bins[-1] = 1.0
        bin_ids = np.digitize(confidence, bins[1:-1], right=True)
        x_values = []
        y_values = []
        for bin_index in range(10):
            mask = bin_ids == bin_index
            if not np.any(mask):
                continue
            x_values.append(float(np.mean(confidence[mask])))
            y_values.append(float(np.mean(correctness[mask])))
        if x_values:
            ax.plot(x_values, y_values, marker="o", label=split_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("Confidence Calibration Curves")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2), max(5, len(class_names) * 1.0)))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_title(title)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, str(matrix[row_index, col_index]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


class DenseEmbeddingDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        feature_row = self.features[index]
        label = None if self.labels is None else int(self.labels[index])
        weight = None if self.sample_weights is None else float(self.sample_weights[index])
        return feature_row, label, weight


def collate_dense_batch(batch):
    features = np.stack([item[0] for item in batch], axis=0).astype(np.float32) / INT8_SCALE
    feature_tensor = torch.from_numpy(features)
    labels = None
    if batch and batch[0][1] is not None:
        labels = torch.as_tensor([int(item[1]) for item in batch], dtype=torch.long)
    weights = None
    if batch and batch[0][2] is not None:
        weights = torch.as_tensor([float(item[2]) for item in batch], dtype=torch.float32)
    return feature_tensor, labels, weights


class MultinomialLogReg(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


class DenseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("MLP requires at least one hidden dimension.")
        layers: list[nn.Module] = []
        previous_dim = input_dim
        self.input_layer = nn.Linear(previous_dim, hidden_dims[0])
        layers.extend([self.input_layer, nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        previous_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class RoleAwareTransformerClassifier(nn.Module):
    def __init__(
        self,
        role_dim: int,
        num_classes: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("transformer-d-model must be divisible by transformer-num-heads.")
        self.role_dim = role_dim
        self.input_proj = nn.Linear(role_dim, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.role_embedding = nn.Parameter(torch.zeros(1, 4, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.role_embedding, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        roles = inputs.view(batch_size, 3, self.role_dim)
        roles = self.input_proj(roles)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, roles], dim=1)
        tokens = tokens + self.role_embedding[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded[:, 0])
        return self.head(pooled)


def create_model(args: argparse.Namespace, input_dim: int, num_classes: int) -> nn.Module:
    if args.model_type == "logreg":
        return MultinomialLogReg(input_dim=input_dim, num_classes=num_classes)
    if args.model_type == "mlp":
        return DenseMLP(
            input_dim=input_dim,
            hidden_dims=parse_int_list(args.hidden_dims),
            num_classes=num_classes,
            dropout=args.dropout,
        )
    if args.model_type == "role-transformer":
        return RoleAwareTransformerClassifier(
            role_dim=args.embedding_prefix_dim,
            num_classes=num_classes,
            model_dim=args.transformer_d_model,
            num_heads=args.transformer_num_heads,
            num_layers=args.transformer_num_layers,
            ff_mult=args.transformer_ff_mult,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported model type: {args.model_type}")


def resolve_learning_rate(args: argparse.Namespace) -> float:
    if args.learning_rate is not None:
        return float(args.learning_rate)
    if args.model_type == "logreg":
        return 5e-3
    if args.model_type == "role-transformer":
        return 5e-4
    return 1e-3


def resolve_device(device_name: str) -> torch.device:
    requested = device_name.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return torch.device(requested)


def create_loader(
    features: np.ndarray,
    labels: np.ndarray | None,
    sample_weights: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = DenseEmbeddingDataset(features=features, labels=labels, sample_weights=sample_weights)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_dense_batch,
    )


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, reduction="none")
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.clamp_min(MIN_SAMPLE_WEIGHT)
    return (losses * weights).sum() / weights.sum()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for features, labels, weights in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = None if weights is None else weights.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = weighted_cross_entropy(logits, labels, weights)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray | None, np.ndarray, float | None]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    for features, labels, _weights in loader:
        features = features.to(device, non_blocking=True)
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probabilities)

        if labels is not None:
            labels_device = labels.to(device, non_blocking=True)
            loss = F.cross_entropy(logits, labels_device, reduction="mean")
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            all_labels.append(labels.cpu().numpy())

    y_true = None if not all_labels else np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    mean_loss = None if total_count == 0 else total_loss / total_count
    return y_true, y_prob, mean_loss


def selection_score(metrics: dict[str, float | None], metric_name: str) -> float:
    if metric_name == "neg_logloss":
        value = metrics.get("logloss")
        return float("-inf") if value is None else -float(value)
    value = metrics.get(metric_name)
    return float("-inf") if value is None else float(value)


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def write_training_state(
    output_path: Path,
    epoch: int,
    history: dict[str, list[float]],
    best_metric_name: str,
    best_metric_value: float,
    best_epoch: int,
    latest_checkpoint_path: Path | None,
    best_model_path: Path | None,
    last_model_path: Path | None,
) -> None:
    payload = {
        "updated_at": utc_now(),
        "epoch": epoch,
        "history": history,
        "selection_metric": best_metric_name,
        "best_metric_value": safe_round(best_metric_value),
        "best_epoch": best_epoch,
        "latest_checkpoint_path": None if latest_checkpoint_path is None else str(latest_checkpoint_path.resolve()),
        "best_model_path": None if best_model_path is None else str(best_model_path.resolve()),
        "last_model_path": None if last_model_path is None else str(last_model_path.resolve()),
    }
    write_json(output_path, payload)


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: dict[str, list[float]],
    best_metric_name: str,
    best_metric_value: float,
    best_epoch: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "selection_metric": best_metric_name,
            "best_metric_value": best_metric_value,
            "best_epoch": best_epoch,
        },
        output_path,
    )


def save_feature_importance(
    model: nn.Module,
    model_type: str,
    feature_names: list[str],
    output_dir: Path,
    top_k: int,
) -> pd.DataFrame:
    if model_type == "logreg":
        weights = model.classifier.weight.detach().cpu().numpy()
        importance = np.abs(weights).mean(axis=0)
        method = "mean_abs_logit_weight"
    elif model_type == "role-transformer":
        projection_weights = model.input_proj.weight.detach().cpu().numpy()
        role_importance = np.abs(projection_weights).mean(axis=0)
        if len(feature_names) % len(role_importance) != 0:
            raise ValueError(
                "Role-transformer feature importance could not be aligned to feature columns: "
                f"{len(feature_names)} columns vs {len(role_importance)} role dimensions."
            )
        repeat_factor = len(feature_names) // len(role_importance)
        importance = np.tile(role_importance, repeat_factor)
        method = "mean_abs_role_projection_weight"
    else:
        first_layer_weights = model.input_layer.weight.detach().cpu().numpy()
        importance = np.abs(first_layer_weights).mean(axis=0)
        method = "mean_abs_first_layer_weight"

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance.astype(np.float64),
            "importance_method": method,
        }
    ).sort_values("importance", ascending=False)
    frame.to_csv(output_dir / "feature_importance.csv", index=False)

    top_frame = frame.head(top_k).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_frame) * 0.35)))
    ax.barh(top_frame["feature"], top_frame["importance"], color="#1f77b4")
    ax.set_title(f"Top {len(top_frame)} Input Weights")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_topk.png", dpi=150)
    plt.close(fig)
    return frame


def training_summary(
    args: argparse.Namespace,
    feature_columns: list[str],
    metrics_by_split: dict[str, dict],
    model_path: Path,
    checkpoint_path: Path | None,
    importance_path: Path,
    class_names: list[str],
    class_weight_map: dict[str, float] | None,
    weight_sanitization: dict[str, dict],
    best_epoch: int,
    resolved_learning_rate: float,
    device_name: str,
    coldstart_zeroed_rows: dict[str, int],
    label_order: list[str],
) -> dict:
    merged_label_map = None
    if args.label_mode == "merged3":
        merged_label_map = MERGED_THREE_LABEL_MAP
    elif args.label_mode == "merged2":
        merged_label_map = MERGED_TWO_LABEL_MAP
    return {
        "created_at": utc_now(),
        "target_column": args.target_col,
        "weight_column": args.weight_col,
        "training_weight_policy": "class-balanced loss weighting with optional label_weight",
        "evaluation_weight_policy": "unweighted metrics and unweighted validation early stopping",
        "feature_source": "raw query/user/candidate embedding prefixes",
        "store_dir": str(Path(args.store_dir).resolve()),
        "label_mode": args.label_mode,
        "merged_label_map": merged_label_map,
        "label_order": label_order,
        "label_order_policy": "explicit preferred order with sorted fallback for unseen labels",
        "model_type": args.model_type,
        "embedding_prefix_dim": args.embedding_prefix_dim,
        "input_scaling": "int8 embeddings divided by 127.0 before modeling",
        "coldstart_user_embedding_policy": "user embedding prefix is zeroed when embedding_user_available == 0 or is_coldstart == 1",
        "coldstart_zeroed_rows": coldstart_zeroed_rows,
        "class_names": class_names,
        "num_classes": len(class_names),
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "checkpoint_interval": args.checkpoint_interval,
        "batch_size": args.batch_size,
        "learning_rate": safe_round(resolved_learning_rate),
        "weight_decay": safe_round(args.weight_decay),
        "dropout": safe_round(args.dropout),
        "hidden_dims": parse_int_list(args.hidden_dims) if args.model_type == "mlp" else [],
        "transformer_d_model": int(args.transformer_d_model),
        "transformer_num_heads": int(args.transformer_num_heads),
        "transformer_num_layers": int(args.transformer_num_layers),
        "transformer_ff_mult": int(args.transformer_ff_mult),
        "selection_metric": args.selection_metric,
        "best_epoch": int(best_epoch),
        "device": device_name,
        "class_imbalance_handling": args.class_imbalance_handling,
        "class_weight_map": class_weight_map,
        "weight_sanitization": weight_sanitization,
        "feature_count": len(feature_columns),
        "feature_columns_path": str((Path(args.run_dir) / "artifacts" / "feature_columns.json").resolve()),
        "model_path": str(model_path.resolve()),
        "latest_checkpoint_path": None if checkpoint_path is None else str(checkpoint_path.resolve()),
        "feature_importance_path": str(importance_path.resolve()),
        "metrics": metrics_by_split,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate dense classifiers on raw query/user/candidate embedding prefixes."
    )
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--weight-col", default="label_weight")
    parser.add_argument("--metadata-cols", default=DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--label-mode", default="original", choices=["original", "merged3", "merged2"])
    parser.add_argument("--merged-training", action="store_true")
    parser.add_argument("--label-order", default=DEFAULT_LABEL_ORDER)
    parser.add_argument("--model-type", default="logreg", choices=["logreg", "mlp", "role-transformer"])
    parser.add_argument("--embedding-prefix-dim", type=positive_int, default=1024)
    parser.add_argument("--batch-size", type=positive_int, default=2048)
    parser.add_argument("--max-epochs", type=positive_int, default=40)
    parser.add_argument("--early-stopping-patience", type=positive_int, default=6)
    parser.add_argument("--checkpoint-interval", type=positive_int, default=2)
    parser.add_argument("--learning-rate", type=positive_float, default=None)
    parser.add_argument("--weight-decay", type=non_negative_float, default=1e-4)
    parser.add_argument("--dropout", type=non_negative_float, default=0.1)
    parser.add_argument("--hidden-dims", default="1024,512")
    parser.add_argument("--transformer-d-model", type=positive_int, default=512)
    parser.add_argument("--transformer-num-heads", type=positive_int, default=8)
    parser.add_argument("--transformer-num-layers", type=positive_int, default=3)
    parser.add_argument("--transformer-ff-mult", type=positive_int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=non_negative_int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-top-k", type=positive_int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--selection-metric",
        default="f1_macro",
        choices=["f1_macro", "balanced_accuracy", "accuracy", "neg_logloss"],
    )
    parser.add_argument(
        "--class-imbalance-handling",
        default="off",
        choices=["off", "balanced-sample-weight"],
        help="Turn class imbalance handling off or use per-class balanced sample weights derived from the train split.",
    )
    return parser


def main() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "This trainer requires PyTorch for both logistic regression and MLP dense-vector models."
        ) from TORCH_IMPORT_ERROR

    parser = build_parser()
    args = parser.parse_args()

    if args.merged_training:
        args.label_mode = "merged3"
    if args.model_type == "logreg":
        args.dropout = 0.0

    set_random_seed(args.seed)
    device = resolve_device(args.device)

    feature_dir = Path(args.feature_dir)
    run_dir = Path(args.run_dir)
    artifacts_dir = run_dir / "artifacts"
    plots_dir = run_dir / "plots"
    predictions_dir = run_dir / "predictions"
    checkpoints_dir = run_dir / "checkpoints"
    models_dir = run_dir / "models"
    for path in (run_dir, artifacts_dir, plots_dir, predictions_dir, checkpoints_dir, models_dir):
        ensure_dir(path)

    checkpoint_state_path = checkpoints_dir / "training_state.json"
    best_model_path = models_dir / "model_best.pt"
    last_model_path = models_dir / "model_last.pt"

    train_df = read_split(feature_dir, "train")
    valid_df = read_split(feature_dir, "valid")
    test_df = read_split(feature_dir, "test")
    embedding_store = load_embedding_store(Path(args.store_dir))

    metadata_columns = parse_csv_list(args.metadata_cols)
    raw_frames = {"train": train_df, "valid": valid_df, "test": test_df}

    features_by_split: dict[str, np.ndarray] = {}
    labels_by_split: dict[str, np.ndarray | None] = {}
    weights_by_split: dict[str, np.ndarray | None] = {}
    metadata_by_split: dict[str, pd.DataFrame] = {}
    coldstart_zeroed_rows: dict[str, int] = {}

    for split_name, frame in raw_frames.items():
        for required_column in EMBEDDING_ID_COLUMNS:
            if required_column not in frame.columns:
                raise ValueError(f"Missing required embedding ID column '{required_column}' in split '{split_name}'.")
        labels, weights, metadata = split_labels_and_metadata(
            frame,
            target_column=args.target_col,
            weight_column=args.weight_col,
            metadata_columns=metadata_columns,
        )
        features, zeroed_count = build_raw_embedding_feature_matrix(
            frame,
            embedding_store=embedding_store,
            embedding_prefix_dim=args.embedding_prefix_dim,
        )
        features_by_split[split_name] = features
        labels_by_split[split_name] = labels
        weights_by_split[split_name] = weights
        metadata_by_split[split_name] = metadata
        coldstart_zeroed_rows[split_name] = zeroed_count

    original_labels_by_split = labels_by_split.copy()
    labels_by_split = {split_name: merge_labels(labels, args.label_mode) for split_name, labels in labels_by_split.items()}

    if labels_by_split["train"] is None or labels_by_split["valid"] is None:
        raise ValueError("Train and valid splits must contain the target column.")

    if args.label_mode == "merged3":
        label_order = parse_csv_list(DEFAULT_MERGED_LABEL_ORDER)
        merged_label_map = MERGED_THREE_LABEL_MAP
    elif args.label_mode == "merged2":
        label_order = parse_csv_list(DEFAULT_BINARY_LABEL_ORDER)
        merged_label_map = MERGED_TWO_LABEL_MAP
    else:
        label_order = parse_csv_list(args.label_order)
        merged_label_map = None
    class_names = infer_label_classes(labels_by_split["train"], preferred_order=label_order)
    class_to_index = {label: index for index, label in enumerate(class_names)}
    encoded_labels_by_split = {
        split_name: encode_labels(labels, class_to_index) for split_name, labels in labels_by_split.items()
    }

    class_weight_vector = compute_class_weight_vector(
        encoded_labels_by_split["train"], len(class_names), args.class_imbalance_handling
    )
    class_weight_map = (
        None
        if class_weight_vector is None
        else {class_name: safe_round(class_weight_vector[index]) for index, class_name in enumerate(class_names)}
    )

    training_weights_by_split = {
        split_name: combine_sample_weights(weights_by_split[split_name], encoded_labels_by_split[split_name], class_weight_vector)
        for split_name in raw_frames
    }
    sanitized_training_weights_by_split: dict[str, np.ndarray | None] = {}
    weight_sanitization: dict[str, dict] = {}
    for split_name, weights in training_weights_by_split.items():
        sanitized_training_weights_by_split[split_name], weight_sanitization[split_name] = sanitize_positive_weights(weights)

    feature_columns = generate_feature_columns(args.embedding_prefix_dim)
    write_json(artifacts_dir / "feature_columns.json", {"feature_columns": feature_columns})
    write_json(
        artifacts_dir / "label_mapping.json",
        {
            "class_names": class_names,
            "class_to_index": class_to_index,
            "label_mode": args.label_mode,
            "merged_label_map": merged_label_map,
            "label_order": label_order,
            "label_order_policy": "explicit preferred order with sorted fallback for unseen labels",
            "class_weight_map": class_weight_map,
        },
    )
    write_json(
        artifacts_dir / "preprocessing.json",
        {
            "created_at": utc_now(),
            "metadata_columns": metadata_columns,
            "feature_source": "raw query/user/candidate embedding prefixes",
            "embedding_prefix_dim": args.embedding_prefix_dim,
            "feature_count": len(feature_columns),
            "label_mode": args.label_mode,
            "merged_label_map": merged_label_map,
            "label_order": label_order,
            "input_scaling": "int8 embeddings divided by 127.0",
            "coldstart_user_embedding_policy": "user embedding prefix is zeroed when embedding_user_available == 0 or is_coldstart == 1",
            "coldstart_zeroed_rows": coldstart_zeroed_rows,
            "class_names": class_names,
            "weight_sanitization": weight_sanitization,
        },
    )

    input_dim = int(features_by_split["train"].shape[1])
    model = create_model(args, input_dim=input_dim, num_classes=len(class_names)).to(device)
    learning_rate = resolve_learning_rate(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    pin_memory = device.type == "cuda"
    train_loader = create_loader(
        features_by_split["train"],
        encoded_labels_by_split["train"],
        sanitized_training_weights_by_split["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    train_eval_loader = create_loader(
        features_by_split["train"],
        encoded_labels_by_split["train"],
        None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = create_loader(
        features_by_split["valid"],
        encoded_labels_by_split["valid"],
        None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = create_loader(
        features_by_split["test"],
        encoded_labels_by_split["test"],
        None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "valid_balanced_accuracy": [],
        "valid_precision_macro": [],
        "valid_recall_macro": [],
        "valid_f1_macro": [],
        "valid_logloss": [],
        "valid_roc_auc_ovr_macro": [],
        "valid_pr_auc_ovr_macro": [],
    }
    start_epoch = 1
    best_metric_value = float("-inf")
    best_epoch = 0

    if args.resume:
        checkpoint_path = latest_checkpoint(checkpoints_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"--resume was requested, but no checkpoint model was found in {checkpoints_dir.resolve()}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = checkpoint.get("history", history)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric_value = float(checkpoint.get("best_metric_value", float("-inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))

    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_true, valid_prob, valid_loss = predict_loader(model, valid_loader, device)
        if valid_true is None:
            raise ValueError("Validation split is missing labels, but labels are required for early stopping.")
        valid_metrics = evaluate_split(valid_true, valid_prob, sample_weight=None, class_names=class_names)
        metric_value = selection_score(valid_metrics, args.selection_metric)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["valid_loss"].append(float(valid_loss if valid_loss is not None else np.nan))
        for key in (
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "logloss",
            "roc_auc_ovr_macro",
            "pr_auc_ovr_macro",
        ):
            history[f"valid_{key}"].append(float(valid_metrics[key]) if valid_metrics[key] is not None else np.nan)

        improved = metric_value > best_metric_value
        if improved:
            best_metric_value = metric_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "selection_metric": args.selection_metric,
                    "best_metric_value": best_metric_value,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        torch.save({"epoch": epoch, "model_state": model.state_dict()}, last_model_path)

        checkpoint_path = None
        if epoch % args.checkpoint_interval == 0 or improved or epoch == args.max_epochs:
            checkpoint_path = checkpoints_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                best_metric_name=args.selection_metric,
                best_metric_value=best_metric_value,
                best_epoch=best_epoch,
            )

        write_training_state(
            checkpoint_state_path,
            epoch=epoch,
            history=history,
            best_metric_name=args.selection_metric,
            best_metric_value=best_metric_value,
            best_epoch=best_epoch,
            latest_checkpoint_path=checkpoint_path or latest_checkpoint(checkpoints_dir),
            best_model_path=best_model_path if best_model_path.exists() else None,
            last_model_path=last_model_path if last_model_path.exists() else None,
        )

        if epochs_without_improvement >= args.early_stopping_patience:
            break

    if best_model_path.exists():
        best_state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_state["model_state"])
    elif last_model_path.exists():
        last_state = torch.load(last_model_path, map_location=device)
        model.load_state_dict(last_state["model_state"])

    metrics_by_split: dict[str, dict] = {}
    labeled_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    split_loaders = {"train": train_eval_loader, "valid": valid_loader, "test": test_loader}
    for split_name, loader in split_loaders.items():
        y_true, y_prob, _loss = predict_loader(model, loader, device)
        save_prediction_frame(
            predictions_dir / f"{split_name}_predictions.parquet",
            metadata_by_split[split_name],
            y_true=y_true,
            y_true_labels=labels_by_split[split_name],
            original_y_true_labels=original_labels_by_split[split_name],
            sample_weight=weights_by_split[split_name],
            effective_sample_weight=sanitized_training_weights_by_split[split_name] if split_name == "train" else None,
            y_prob=y_prob,
            class_names=class_names,
        )
        if y_true is not None:
            metrics_by_split[split_name] = evaluate_split(y_true, y_prob, sample_weight=None, class_names=class_names)
            labeled_predictions[split_name] = (y_true, y_prob)

    save_metrics_table(metrics_by_split, artifacts_dir / "metrics.csv")
    write_json(artifacts_dir / "metrics.json", metrics_by_split)
    save_feature_importance(model, args.model_type, feature_columns, plots_dir, args.plot_top_k)
    plot_learning_curves(history, plots_dir / "learning_curves.png")
    plot_label_distribution(labels_by_split, plots_dir / "label_distribution.png")
    plot_roc_curves({k: v for k, v in labeled_predictions.items() if k in {"valid", "test"}}, plots_dir / "roc_curves.png", len(class_names))
    plot_pr_curves({k: v for k, v in labeled_predictions.items() if k in {"valid", "test"}}, plots_dir / "pr_curves.png", len(class_names))
    plot_prediction_histograms(
        {k: v for k, v in labeled_predictions.items() if k in {"valid", "test"}},
        plots_dir / "prediction_histograms.png",
    )
    plot_calibration_curves(
        {k: v for k, v in labeled_predictions.items() if k in {"valid", "test"}},
        plots_dir / "calibration_curves.png",
    )
    if "valid" in labeled_predictions:
        valid_pred = np.argmax(labeled_predictions["valid"][1], axis=1)
        plot_confusion_matrix(
            labeled_predictions["valid"][0],
            valid_pred,
            class_names=class_names,
            output_path=plots_dir / "confusion_matrix_valid.png",
            title="Validation Confusion Matrix",
        )
    if "test" in labeled_predictions:
        test_pred = np.argmax(labeled_predictions["test"][1], axis=1)
        plot_confusion_matrix(
            labeled_predictions["test"][0],
            test_pred,
            class_names=class_names,
            output_path=plots_dir / "confusion_matrix_test.png",
            title="Test Confusion Matrix",
        )

    final_model_path = best_model_path if best_model_path.exists() else last_model_path
    summary = training_summary(
        args=args,
        feature_columns=feature_columns,
        metrics_by_split=metrics_by_split,
        model_path=final_model_path,
        checkpoint_path=latest_checkpoint(checkpoints_dir),
        importance_path=plots_dir / "feature_importance.csv",
        class_names=class_names,
        class_weight_map=class_weight_map,
        weight_sanitization=weight_sanitization,
        best_epoch=best_epoch,
        resolved_learning_rate=learning_rate,
        device_name=str(device),
        coldstart_zeroed_rows=coldstart_zeroed_rows,
        label_order=label_order,
    )
    write_json(artifacts_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
