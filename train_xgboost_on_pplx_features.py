import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
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


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = str(BASE_DIR / "output" / "feature_splits")
DEFAULT_RUN_DIR = str(BASE_DIR / "training_runs" / "xgboost_pplx")
DEFAULT_FEATURE_COLUMNS = (
    "pplx_qc_cosine_256,pplx_uc_cosine_256,pplx_qu_cosine_256,pplx_quc_cosine_256,"
    "pplx_qc_cosine_512,pplx_uc_cosine_512,pplx_qu_cosine_512,pplx_quc_cosine_512,"
    "pplx_qc_cosine_1024,pplx_uc_cosine_1024,pplx_qu_cosine_1024,pplx_quc_cosine_1024"
)
DEFAULT_DROP_COLUMNS = (
    "event_id,user_id,search_id,profil_id,user_profile_text,query_text,candidate_text,"
    "reason_codes,split"
)
DEFAULT_METADATA_COLUMNS = "event_id,user_id,search_id,profil_id,split"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or parsed > 1:
        raise argparse.ArgumentTypeError("Expected a value in (0, 1].")
    return parsed


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def sanitize_metric_history(evals_result: dict) -> dict[str, dict[str, list[float]]]:
    sanitized: dict[str, dict[str, list[float]]] = {}
    for dataset_name, metrics in evals_result.items():
        sanitized[dataset_name] = {}
        for metric_name, values in metrics.items():
            sanitized[dataset_name][metric_name] = [float(value) for value in values]
    return sanitized


def merge_histories(previous: dict | None, current: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    if not previous:
        return current
    merged = json.loads(json.dumps(previous))
    for dataset_name, metrics in current.items():
        merged.setdefault(dataset_name, {})
        for metric_name, values in metrics.items():
            merged[dataset_name].setdefault(metric_name, [])
            merged[dataset_name][metric_name].extend(float(value) for value in values)
    return merged


def infer_checkpoint_rounds_from_booster(booster) -> int:
    if hasattr(booster, "num_boosted_rounds"):
        return int(booster.num_boosted_rounds())
    return len(booster.get_dump())


def infer_checkpoint_rounds(model: xgb.XGBClassifier) -> int:
    return infer_checkpoint_rounds_from_booster(model.get_booster())


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoint_paths = sorted(checkpoint_dir.glob("model_round_*.json"))
    if not checkpoint_paths:
        return None
    return checkpoint_paths[-1]


class PeriodicCheckpointCallback(xgb.callback.TrainingCallback):
    def __init__(self, checkpoint_dir: Path, state_path: Path, interval: int) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.state_path = state_path
        self.interval = interval

    def _save_state(self, model, epoch: int, evals_log: dict) -> None:
        checkpoint_path = self.checkpoint_dir / f"model_round_{epoch:05d}.json"
        model.save_model(str(checkpoint_path))
        state = {
            "updated_at": utc_now(),
            "checkpoint_round": int(epoch),
            "checkpoint_model_path": str(checkpoint_path.resolve()),
            "evals_result": sanitize_metric_history(evals_log),
        }
        write_json(self.state_path, state)

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        iteration = epoch + 1
        if iteration % self.interval == 0:
            self._save_state(model, iteration, evals_log)
        return False

    def after_training(self, model):
        iteration = infer_checkpoint_rounds_from_booster(model)
        self._save_state(model, iteration, {})
        return model


def read_split(feature_dir: Path, split_name: str) -> pd.DataFrame:
    path = feature_dir / f"{split_name}_with_pplx_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature split: {path}")
    return pd.read_parquet(path)


def engineer_time_features(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    if time_column not in df.columns:
        return df
    timestamp = pd.to_datetime(df[time_column], errors="coerce", utc=True)
    out = df.copy()
    out[f"{time_column}_hour"] = timestamp.dt.hour.astype("float32")
    out[f"{time_column}_dayofweek"] = timestamp.dt.dayofweek.astype("float32")
    out[f"{time_column}_day"] = timestamp.dt.day.astype("float32")
    out[f"{time_column}_month"] = timestamp.dt.month.astype("float32")
    out[f"{time_column}_is_weekend"] = timestamp.dt.dayofweek.isin([5, 6]).astype("float32")
    epoch_seconds = pd.Series(np.nan, index=timestamp.index, dtype="float64")
    valid_mask = timestamp.notna()
    if valid_mask.any():
        epoch_seconds.loc[valid_mask] = timestamp.loc[valid_mask].astype("int64") / 1_000_000_000.0
    out[f"{time_column}_epoch_seconds"] = epoch_seconds.astype("float64")
    return out.drop(columns=[time_column])


def select_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Requested feature columns are missing from the dataset: {missing_columns}")
    return df[feature_columns].copy()


def split_labels_and_features(
    df: pd.DataFrame,
    target_column: str,
    weight_column: str | None,
    metadata_columns: list[str],
    drop_columns: list[str],
    selected_feature_columns: list[str],
    time_column: str,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None, pd.DataFrame]:
    labels = df[target_column].to_numpy() if target_column in df.columns else None
    weights = df[weight_column].to_numpy(dtype=np.float32) if weight_column and weight_column in df.columns else None
    metadata = df[[column for column in metadata_columns if column in df.columns]].copy()

    removable = [column for column in [target_column, weight_column] if column and column in df.columns]
    feature_frame = df.drop(columns=removable)
    feature_frame = engineer_time_features(feature_frame, time_column=time_column)

    existing_drop_columns = [column for column in drop_columns if column in feature_frame.columns]
    if existing_drop_columns:
        feature_frame = feature_frame.drop(columns=existing_drop_columns)

    if selected_feature_columns:
        feature_frame = select_feature_columns(feature_frame, selected_feature_columns)

    return feature_frame, labels, weights, metadata


def one_hot_encode_splits(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    tagged_frames = []
    for split_name, frame in frames.items():
        tagged = frame.copy()
        tagged["__split_name__"] = split_name
        tagged_frames.append(tagged)

    combined = pd.concat(tagged_frames, axis=0, ignore_index=True, sort=False)
    categorical_columns = []
    for column in combined.columns:
        if column == "__split_name__":
            continue
        if (
            pd.api.types.is_object_dtype(combined[column])
            or pd.api.types.is_string_dtype(combined[column])
            or pd.api.types.is_categorical_dtype(combined[column])
            or pd.api.types.is_bool_dtype(combined[column])
        ):
            categorical_columns.append(column)

    for column in categorical_columns:
        combined[column] = combined[column].fillna("__missing__").astype(str)

    encoded = pd.get_dummies(combined, columns=categorical_columns, dummy_na=False)
    split_names = encoded.pop("__split_name__")
    encoded = encoded.astype(np.float32)

    result: dict[str, pd.DataFrame] = {}
    for split_name in frames:
        result[split_name] = encoded.loc[split_names == split_name].reset_index(drop=True)
    return result, encoded.columns.tolist(), categorical_columns


def infer_label_classes(y: np.ndarray) -> list[str]:
    values = pd.Series(y).dropna().astype(str)
    unique_values = sorted(values.unique().tolist())
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
    return series.map(class_to_index).to_numpy(dtype=np.int32)


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


def safe_metric(fn, *args, **kwargs) -> float | None:
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return None


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
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
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
    sample_weight: np.ndarray | None,
    y_prob: np.ndarray,
    class_names: list[str],
) -> None:
    frame = metadata.copy()
    if y_true is not None:
        frame["label_id"] = y_true
    if y_true_labels is not None:
        frame["label"] = y_true_labels
    if sample_weight is not None:
        frame["label_weight"] = sample_weight
    predicted_indices = np.argmax(y_prob, axis=1).astype(np.int32)
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


def save_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    output_dir: Path,
    top_k: int,
) -> pd.DataFrame:
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    cover_scores = booster.get_score(importance_type="cover")
    records = []
    for feature_name in feature_names:
        records.append(
            {
                "feature": feature_name,
                "gain": float(gain_scores.get(feature_name, 0.0)),
                "weight": float(weight_scores.get(feature_name, 0.0)),
                "cover": float(cover_scores.get(feature_name, 0.0)),
            }
        )
    importance_frame = pd.DataFrame(records).sort_values(["gain", "weight"], ascending=False)
    importance_frame.to_csv(output_dir / "feature_importance.csv", index=False)

    top_gain = importance_frame.head(top_k).sort_values("gain")
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_gain) * 0.35)))
    ax.barh(top_gain["feature"], top_gain["gain"], color="#1f77b4")
    ax.set_title(f"Top {len(top_gain)} Feature Importances (Gain)")
    ax.set_xlabel("Gain")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_gain_topk.png", dpi=150)
    plt.close(fig)

    top_weight = importance_frame.head(top_k).sort_values("weight")
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_weight) * 0.35)))
    ax.barh(top_weight["feature"], top_weight["weight"], color="#ff7f0e")
    ax.set_title(f"Top {len(top_weight)} Feature Importances (Weight)")
    ax.set_xlabel("Split Count")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_weight_topk.png", dpi=150)
    plt.close(fig)

    return importance_frame


def plot_learning_curves(evals_result: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    if not evals_result:
        return
    metric_names = sorted({metric for dataset_metrics in evals_result.values() for metric in dataset_metrics})
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, max(4, 3.5 * len(metric_names))), sharex=True)
    if len(metric_names) == 1:
        axes = [axes]
    for axis, metric_name in zip(axes, metric_names):
        for dataset_name, dataset_metrics in evals_result.items():
            if metric_name not in dataset_metrics:
                continue
            x_axis = np.arange(1, len(dataset_metrics[metric_name]) + 1)
            axis.plot(x_axis, dataset_metrics[metric_name], label=dataset_name)
        axis.set_title(metric_name)
        axis.set_ylabel(metric_name)
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Boosting Round")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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


def plot_roc_curves(predictions: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path, num_classes: int) -> None:
    if not predictions:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False
    for split_name, (y_true, y_prob) in predictions.items():
        if len(np.unique(y_true)) < 2:
            continue
        y_bin = label_binarize(y_true, classes=np.arange(num_classes))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        auc_value = roc_auc_score(y_bin, y_prob, average="micro", multi_class="ovr")
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
    for split_name, (y_true, y_prob) in predictions.items():
        y_bin = label_binarize(y_true, classes=np.arange(num_classes))
        precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_prob.ravel())
        ap_value = average_precision_score(y_bin, y_prob, average="micro")
        ax.plot(recall, precision, label=f"{split_name} micro-OVR (AP={ap_value:.4f})")
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


def build_classifier(args: argparse.Namespace, n_estimators: int, num_classes: int) -> xgb.XGBClassifier:
    classifier_kwargs = {
        "n_estimators": n_estimators,
        "objective": "binary:logistic" if num_classes == 2 else "multi:softprob",
        "eval_metric": ["logloss", "auc", "aucpr"] if num_classes == 2 else ["mlogloss", "merror"],
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "tree_method": args.tree_method,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
    }
    if args.device:
        classifier_kwargs["device"] = args.device
    if num_classes > 2:
        classifier_kwargs["num_class"] = num_classes
    return xgb.XGBClassifier(**classifier_kwargs)


def training_summary(
    args: argparse.Namespace,
    feature_columns: list[str],
    categorical_columns: list[str],
    metrics_by_split: dict[str, dict],
    model_path: Path,
    checkpoint_path: Path | None,
    importance_path: Path,
    class_names: list[str],
    class_weight_map: dict[str, float] | None,
) -> dict:
    return {
        "created_at": utc_now(),
        "target_column": args.target_col,
        "weight_column": args.weight_col,
        "class_names": class_names,
        "num_classes": len(class_names),
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "checkpoint_interval": args.checkpoint_interval,
        "class_imbalance_handling": args.class_imbalance_handling,
        "class_weight_map": class_weight_map,
        "selected_feature_columns": parse_csv_list(args.feature_cols),
        "drop_columns": parse_csv_list(args.drop_cols),
        "time_column": args.time_col,
        "feature_count": len(feature_columns),
        "feature_columns_path": str((Path(args.run_dir) / "artifacts" / "feature_columns.json").resolve()),
        "categorical_columns": categorical_columns,
        "model_path": str(model_path.resolve()),
        "latest_checkpoint_path": None if checkpoint_path is None else str(checkpoint_path.resolve()),
        "feature_importance_path": str(importance_path.resolve()),
        "metrics": metrics_by_split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an XGBoost classifier on output/feature_splits parquet files."
    )
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--target-col", default="label")
    parser.add_argument("--weight-col", default="label_weight")
    parser.add_argument("--feature-cols", default=DEFAULT_FEATURE_COLUMNS)
    parser.add_argument("--drop-cols", default=DEFAULT_DROP_COLUMNS)
    parser.add_argument("--metadata-cols", default=DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--time-col", default="searched_at")
    parser.add_argument("--num-boost-round", type=positive_int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=positive_int, default=100)
    parser.add_argument("--checkpoint-interval", type=positive_int, default=50)
    parser.add_argument("--learning-rate", type=positive_float, default=0.03)
    parser.add_argument("--max-depth", type=positive_int, default=8)
    parser.add_argument("--min-child-weight", type=positive_float, default=1.0)
    parser.add_argument("--subsample", type=probability, default=0.9)
    parser.add_argument("--colsample-bytree", type=probability, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=positive_float, default=1.0)
    parser.add_argument("--tree-method", default="hist", choices=["hist", "approx", "exact"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-top-k", type=positive_int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--class-imbalance-handling",
        default="off",
        choices=["off", "balanced-sample-weight"],
        help="Turn class imbalance handling off or use per-class balanced sample weights derived from the train split.",
    )
    args = parser.parse_args()

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

    train_df = read_split(feature_dir, "train")
    valid_df = read_split(feature_dir, "valid")
    test_df = read_split(feature_dir, "test")

    metadata_columns = parse_csv_list(args.metadata_cols)
    drop_columns = parse_csv_list(args.drop_cols)
    selected_feature_columns = parse_csv_list(args.feature_cols)

    raw_frames = {"train": train_df, "valid": valid_df, "test": test_df}
    feature_frames: dict[str, pd.DataFrame] = {}
    labels_by_split: dict[str, np.ndarray | None] = {}
    weights_by_split: dict[str, np.ndarray | None] = {}
    metadata_by_split: dict[str, pd.DataFrame] = {}
    for split_name, frame in raw_frames.items():
        features, labels, weights, metadata = split_labels_and_features(
            frame,
            target_column=args.target_col,
            weight_column=args.weight_col,
            metadata_columns=metadata_columns,
            drop_columns=drop_columns,
            selected_feature_columns=selected_feature_columns,
            time_column=args.time_col,
        )
        feature_frames[split_name] = features
        labels_by_split[split_name] = labels
        weights_by_split[split_name] = weights
        metadata_by_split[split_name] = metadata

    if labels_by_split["train"] is None or labels_by_split["valid"] is None:
        raise ValueError("Train and valid splits must contain the target column.")

    class_names = infer_label_classes(labels_by_split["train"])
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

    encoded_frames, feature_columns, categorical_columns = one_hot_encode_splits(feature_frames)
    write_json(artifacts_dir / "feature_columns.json", {"feature_columns": feature_columns})
    write_json(
        artifacts_dir / "label_mapping.json",
        {
            "class_names": class_names,
            "class_to_index": class_to_index,
            "class_weight_map": class_weight_map,
        },
    )
    write_json(
        artifacts_dir / "preprocessing.json",
        {
            "created_at": utc_now(),
            "selected_feature_columns": selected_feature_columns,
            "drop_columns": drop_columns,
            "metadata_columns": metadata_columns,
            "categorical_columns_encoded": categorical_columns,
            "time_column": args.time_col,
            "feature_count": len(feature_columns),
            "class_names": class_names,
        },
    )
    previous_history = None
    loaded_model = None
    completed_rounds = 0

    if args.resume:
        checkpoint_path = latest_checkpoint(checkpoints_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"--resume was requested, but no checkpoint model was found in {checkpoints_dir.resolve()}"
            )
        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(str(checkpoint_path))
        completed_rounds = infer_checkpoint_rounds(loaded_model)
        checkpoint_state = load_json_if_exists(checkpoint_state_path) or {}
        previous_history = checkpoint_state.get("history")

    remaining_rounds = max(args.num_boost_round - completed_rounds, 0)

    classifier = build_classifier(args, max(remaining_rounds, 1), len(class_names))
    callbacks: list = [PeriodicCheckpointCallback(checkpoints_dir, checkpoint_state_path, args.checkpoint_interval)]
    if args.early_stopping_rounds > 0:
        callbacks.append(xgb.callback.EarlyStopping(rounds=args.early_stopping_rounds, save_best=True))

    if remaining_rounds > 0:
        fit_kwargs = {
            "sample_weight": training_weights_by_split["train"],
            "eval_set": [
                (encoded_frames["train"], encoded_labels_by_split["train"]),
                (encoded_frames["valid"], encoded_labels_by_split["valid"]),
            ],
            "verbose": False,
            "callbacks": callbacks,
        }
        if loaded_model is not None:
            fit_kwargs["xgb_model"] = loaded_model
        if training_weights_by_split["train"] is not None or training_weights_by_split["valid"] is not None:
            fit_kwargs["sample_weight_eval_set"] = [training_weights_by_split["train"], training_weights_by_split["valid"]]
        classifier.fit(
            encoded_frames["train"],
            encoded_labels_by_split["train"],
            **fit_kwargs,
        )
    else:
        if loaded_model is None:
            raise ValueError("Training requested zero remaining rounds without a loaded checkpoint.")
        classifier = loaded_model

    current_history = classifier.evals_result() if remaining_rounds > 0 else {}
    merged_history = merge_histories(previous_history, sanitize_metric_history(current_history))
    final_model_path = models_dir / "xgboost_model_final.json"
    classifier.save_model(str(final_model_path))

    final_checkpoint_path = latest_checkpoint(checkpoints_dir)
    current_rounds = infer_checkpoint_rounds(classifier)
    checkpoint_state = {
        "updated_at": utc_now(),
        "completed_rounds": current_rounds,
        "target_rounds": args.num_boost_round,
        "history": merged_history,
        "latest_checkpoint_path": None if final_checkpoint_path is None else str(final_checkpoint_path.resolve()),
        "final_model_path": str(final_model_path.resolve()),
        "best_iteration": int(getattr(classifier, "best_iteration", current_rounds - 1))
        if current_rounds > 0
        else None,
    }
    write_json(checkpoint_state_path, checkpoint_state)

    metrics_by_split: dict[str, dict] = {}
    labeled_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name in ("train", "valid", "test"):
        y_true = encoded_labels_by_split[split_name]
        y_prob = classifier.predict_proba(encoded_frames[split_name])
        save_prediction_frame(
            predictions_dir / f"{split_name}_predictions.parquet",
            metadata_by_split[split_name],
            y_true=y_true,
            y_true_labels=labels_by_split[split_name],
            sample_weight=weights_by_split[split_name],
            y_prob=y_prob,
            class_names=class_names,
        )
        if y_true is not None:
            metrics_by_split[split_name] = evaluate_split(
                y_true,
                y_prob,
                sample_weight=weights_by_split[split_name],
                class_names=class_names,
            )
            labeled_predictions[split_name] = (y_true, y_prob)

    save_metrics_table(metrics_by_split, artifacts_dir / "metrics.csv")
    write_json(artifacts_dir / "metrics.json", metrics_by_split)

    save_feature_importance(classifier, feature_columns, plots_dir, args.plot_top_k)
    plot_learning_curves(merged_history, plots_dir / "learning_curves.png")
    plot_label_distribution(labels_by_split, plots_dir / "label_distribution.png")
    plot_roc_curves(
        {k: v for k, v in labeled_predictions.items() if k in {"valid", "test"}},
        plots_dir / "roc_curves.png",
        len(class_names),
    )
    plot_pr_curves(
        {k: v for k, v in labeled_predictions.items() if k in {"valid", "test"}},
        plots_dir / "pr_curves.png",
        len(class_names),
    )
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

    summary = training_summary(
        args=args,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        metrics_by_split=metrics_by_split,
        model_path=final_model_path,
        checkpoint_path=final_checkpoint_path,
        importance_path=plots_dir / "feature_importance.csv",
        class_names=class_names,
        class_weight_map=class_weight_map,
    )
    write_json(artifacts_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
