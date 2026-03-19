#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FEATURE_DIR="${FEATURE_DIR:-$ROOT_DIR/output/feature_splits}"
STORE_DIR="${STORE_DIR:-$ROOT_DIR/output/store}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/ablation_runs}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
ABLATION_DIR="${ABLATION_DIR:-$RUN_ROOT/$TIMESTAMP}"
RUNS_DIR="$ABLATION_DIR/runs"
LOGS_DIR="$ABLATION_DIR/logs"
REPORTS_DIR="$ABLATION_DIR/reports"
CALIBRATION_DIR="$ABLATION_DIR/calibration_plots"
DEVICE="${DEVICE:-auto}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-6}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-0}"
HIDDEN_DIMS="${HIDDEN_DIMS:-1024,512}"
DROPOUT="${DROPOUT:-0.1}"
MLP_CONFIGS="${MLP_CONFIGS:-1024,512@0.1;1536,768@0.15;2048,1024@0.2}"
TRANSFORMER_CONFIGS="${TRANSFORMER_CONFIGS:-512,8,2,4,0.1;768,8,3,4,0.1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MODEL_TYPES="logreg,role-transformer"
DIMS="256,512,1024"
IMBALANCE_MODES="off,balanced-sample-weight"
LABEL_MODE="${LABEL_MODE:-original}"

usage() {
  cat <<EOF
Usage: bash run_xgboost_ablation.sh [--model-types logreg,role-transformer] [--dims 256,512,1024] [--imbalance-modes off,balanced-sample-weight] [--label-mode original|merged3|merged2] [--transformer-configs 512,8,2,4,0.1;768,8,3,4,0.1]

Options:
  --model-types        Comma-separated dense model families to evaluate. Default: logreg,role-transformer
  --dims               Comma-separated embedding prefix dimensions to evaluate. Default: 256,512,1024
  --imbalance-modes    Comma-separated imbalance modes. Default: off,balanced-sample-weight
  --label-mode         Label target mode. Use original 5 classes, merged3, or merged2.
  --mlp-configs        Semicolon-separated hidden-dim/dropout specs for mlp. Format: hidden1,hidden2@dropout
  --transformer-configs Semicolon-separated transformer specs. Format: d_model,num_heads,num_layers,ff_mult,dropout
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-types)
      MODEL_TYPES="$2"
      shift 2
      ;;
    --dims)
      DIMS="$2"
      shift 2
      ;;
    --imbalance-modes)
      IMBALANCE_MODES="$2"
      shift 2
      ;;
    --label-mode)
      LABEL_MODE="$2"
      shift 2
      ;;
    --mlp-configs)
      MLP_CONFIGS="$2"
      shift 2
      ;;
    --transformer-configs)
      TRANSFORMER_CONFIGS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

IFS=',' read -r -a MODEL_VALUES <<<"$MODEL_TYPES"
IFS=',' read -r -a DIM_VALUES <<<"$DIMS"
IFS=',' read -r -a IMBALANCE_VALUES <<<"$IMBALANCE_MODES"

mkdir -p "$RUNS_DIR" "$LOGS_DIR" "$REPORTS_DIR" "$CALIBRATION_DIR"

MANIFEST_PATH="$REPORTS_DIR/ablation_manifest.tsv"
SUMMARY_CSV="$REPORTS_DIR/ablation_summary.csv"
SUMMARY_MD="$REPORTS_DIR/ablation_summary.md"
CALIBRATION_INDEX="$REPORTS_DIR/calibration_index.md"

cat >"$MANIFEST_PATH" <<EOF
experiment_name	model_type	class_imbalance_handling	embedding_prefix_dim	run_dir	log_path	status
EOF

echo "Ablation root: $ABLATION_DIR"
echo "Using feature dir: $FEATURE_DIR"
echo "Using store dir:   $STORE_DIR"
echo "Using python:      $PYTHON_BIN"
echo "Model types:       $MODEL_TYPES"
echo "Embedding dims:    $DIMS"
echo "Imbalance modes:   $IMBALANCE_MODES"
echo "Label mode:        $LABEL_MODE"
echo "MLP configs:       $MLP_CONFIGS"
echo "Transformer cfgs:  $TRANSFORMER_CONFIGS"

failures=()

run_experiment() {
  local name="$1"
  local model_type="$2"
  local imbalance="$3"
  local prefix_dim="$4"
  local hidden_dims="$5"
  local dropout="$6"
  local transformer_d_model="$7"
  local transformer_num_heads="$8"
  local transformer_num_layers="$9"
  local transformer_ff_mult="${10}"
  local run_dir="$RUNS_DIR/$name"
  local log_path="$LOGS_DIR/$name.log"
  local calibration_target="$CALIBRATION_DIR/${name}_calibration.png"
  local status="success"

  mkdir -p "$run_dir"

  echo
  echo "============================================================"
  echo "Running experiment: $name"
  echo "Model type:        $model_type"
  echo "Class imbalance:   $imbalance"
  echo "Embedding dim:     $prefix_dim"
  if [[ "$model_type" == "mlp" ]]; then
    echo "Hidden dims:       $hidden_dims"
    echo "Dropout:           $dropout"
  elif [[ "$model_type" == "role-transformer" ]]; then
    echo "Transformer d_model: $transformer_d_model"
    echo "Transformer heads:   $transformer_num_heads"
    echo "Transformer layers:  $transformer_num_layers"
    echo "Transformer ff_mult: $transformer_ff_mult"
    echo "Dropout:             $dropout"
  fi
  echo "Run dir:           $run_dir"
  echo "============================================================"

  local cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/train_dense_embedding_classifier.py"
    --feature-dir "$FEATURE_DIR"
    --store-dir "$STORE_DIR"
    --run-dir "$run_dir"
    --model-type "$model_type"
    --label-mode "$LABEL_MODE"
    --embedding-prefix-dim "$prefix_dim"
    --class-imbalance-handling "$imbalance"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --max-epochs "$MAX_EPOCHS"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
  )
  if [[ "$model_type" == "mlp" ]]; then
    cmd+=(--hidden-dims "$hidden_dims" --dropout "$dropout")
  elif [[ "$model_type" == "role-transformer" ]]; then
    cmd+=(
      --transformer-d-model "$transformer_d_model"
      --transformer-num-heads "$transformer_num_heads"
      --transformer-num-layers "$transformer_num_layers"
      --transformer-ff-mult "$transformer_ff_mult"
      --dropout "$dropout"
    )
  fi

  if "${cmd[@]}" \
    ${EXTRA_ARGS} 2>&1 | tee "$log_path"; then
    if [[ -f "$run_dir/plots/calibration_curves.png" ]]; then
      cp "$run_dir/plots/calibration_curves.png" "$calibration_target"
    fi
  else
    status="failed"
    failures+=("$name")
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$model_type" "$imbalance" "$prefix_dim" "$run_dir" "$log_path" "$status" >>"$MANIFEST_PATH"
}

for model_type in "${MODEL_VALUES[@]}"; do
  model_type="$(echo "$model_type" | xargs)"
  [[ -n "$model_type" ]] || continue
  if [[ "$model_type" == "mlp" ]]; then
    IFS=';' read -r -a MLP_CONFIG_VALUES <<<"$MLP_CONFIGS"
    for config in "${MLP_CONFIG_VALUES[@]}"; do
      config="$(echo "$config" | xargs)"
      [[ -n "$config" ]] || continue
      hidden_dims="${config%@*}"
      dropout="${config#*@}"
      config_label="mlp_$(echo "$hidden_dims" | tr ',' 'x')_d$(echo "$dropout" | tr -d '.')"
      for dim in "${DIM_VALUES[@]}"; do
        dim="$(echo "$dim" | xargs)"
        [[ -n "$dim" ]] || continue
        for imbalance in "${IMBALANCE_VALUES[@]}"; do
          imbalance="$(echo "$imbalance" | xargs)"
          [[ -n "$imbalance" ]] || continue
          run_experiment "${config_label}_rawdim${dim}_${imbalance//-/_}" "$model_type" "$imbalance" "$dim" "$hidden_dims" "$dropout" "" "" "" ""
        done
      done
    done
  elif [[ "$model_type" == "role-transformer" ]]; then
    IFS=';' read -r -a TRANSFORMER_CONFIG_VALUES <<<"$TRANSFORMER_CONFIGS"
    for config in "${TRANSFORMER_CONFIG_VALUES[@]}"; do
      config="$(echo "$config" | xargs)"
      [[ -n "$config" ]] || continue
      IFS=',' read -r transformer_d_model transformer_num_heads transformer_num_layers transformer_ff_mult dropout <<<"$config"
      transformer_label="trf_${transformer_d_model}d_${transformer_num_heads}h_${transformer_num_layers}l_ff${transformer_ff_mult}_d$(echo "$dropout" | tr -d '.')"
      for dim in "${DIM_VALUES[@]}"; do
        dim="$(echo "$dim" | xargs)"
        [[ -n "$dim" ]] || continue
        for imbalance in "${IMBALANCE_VALUES[@]}"; do
          imbalance="$(echo "$imbalance" | xargs)"
          [[ -n "$imbalance" ]] || continue
          run_experiment "${transformer_label}_rawdim${dim}_${imbalance//-/_}" "$model_type" "$imbalance" "$dim" "" "$dropout" "$transformer_d_model" "$transformer_num_heads" "$transformer_num_layers" "$transformer_ff_mult"
        done
      done
    done
  else
    for dim in "${DIM_VALUES[@]}"; do
      dim="$(echo "$dim" | xargs)"
      [[ -n "$dim" ]] || continue
      for imbalance in "${IMBALANCE_VALUES[@]}"; do
        imbalance="$(echo "$imbalance" | xargs)"
        [[ -n "$imbalance" ]] || continue
        run_experiment "${model_type}_rawdim${dim}_${imbalance//-/_}" "$model_type" "$imbalance" "$dim" "" "0.0" "" "" "" ""
      done
    done
  fi
done

"$PYTHON_BIN" - "$MANIFEST_PATH" "$SUMMARY_CSV" "$SUMMARY_MD" "$CALIBRATION_INDEX" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
calibration_index = Path(sys.argv[4])

rows = []
with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    rows.extend(reader)

summary_rows = []
for row in rows:
    run_dir = Path(row["run_dir"])
    metrics_path = run_dir / "artifacts" / "metrics.json"
    summary_path = run_dir / "artifacts" / "run_summary.json"
    calibration_path = run_dir / "plots" / "calibration_curves.png"

    record = {
        "experiment_name": row["experiment_name"],
        "status": row["status"],
        "model_type": row["model_type"],
        "class_imbalance_handling": row["class_imbalance_handling"],
        "embedding_prefix_dim": row["embedding_prefix_dim"],
        "run_dir": row["run_dir"],
        "log_path": row["log_path"],
        "calibration_plot": str(calibration_path) if calibration_path.exists() else "",
    }

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for split_name in ("valid", "test", "train"):
            for key, value in metrics.get(split_name, {}).items():
                record[f"{split_name}_{key}"] = value

    if summary_path.exists():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record["feature_count"] = run_summary.get("feature_count")
        record["class_names"] = ",".join(run_summary.get("class_names", []))
        record["hidden_dims"] = ",".join(str(value) for value in run_summary.get("hidden_dims", []))
        record["best_epoch"] = run_summary.get("best_epoch")

    summary_rows.append(record)

summary_rows.sort(
    key=lambda row: (
        float(row.get("valid_f1_macro", float("-inf"))),
        float(row.get("valid_balanced_accuracy", float("-inf"))),
        float(row.get("valid_accuracy", float("-inf"))),
    ),
    reverse=True,
)

fieldnames = [
    "experiment_name",
    "status",
    "model_type",
    "embedding_prefix_dim",
    "class_imbalance_handling",
    "hidden_dims",
    "feature_count",
    "best_epoch",
    "valid_f1_macro",
    "valid_balanced_accuracy",
    "valid_accuracy",
    "valid_logloss",
    "test_f1_macro",
    "test_balanced_accuracy",
    "test_accuracy",
    "test_logloss",
    "class_names",
    "run_dir",
    "log_path",
    "calibration_plot",
]

with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary_rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

with summary_md.open("w", encoding="utf-8") as handle:
    handle.write("# Raw Embedding Dense Model Ablation Summary\n\n")
    if summary_rows:
        best = summary_rows[0]
        handle.write("## Best Validation Run\n\n")
        handle.write(f"- Experiment: `{best['experiment_name']}`\n")
        handle.write(f"- Model type: `{best['model_type']}`\n")
        handle.write(f"- Embedding prefix dim: `{best['embedding_prefix_dim']}`\n")
        handle.write(f"- Class imbalance handling: `{best['class_imbalance_handling']}`\n")
        handle.write(f"- Hidden dims: `{best.get('hidden_dims', '')}`\n")
        handle.write(f"- Validation macro F1: `{best.get('valid_f1_macro', '')}`\n")
        handle.write(f"- Validation balanced accuracy: `{best.get('valid_balanced_accuracy', '')}`\n")
        handle.write(f"- Test macro F1: `{best.get('test_f1_macro', '')}`\n")
        handle.write(f"- Run directory: `{best['run_dir']}`\n\n")
        handle.write("| Experiment | Model | Dim | Imbalance | Valid F1 Macro | Valid Balanced Acc | Test F1 Macro | Test Balanced Acc |\n")
        handle.write("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |\n")
        for row in summary_rows:
            handle.write(
                f"| {row['experiment_name']} | {row['model_type']} | {row['embedding_prefix_dim']} "
                f"| {row['class_imbalance_handling']} | {row.get('valid_f1_macro', '')} "
                f"| {row.get('valid_balanced_accuracy', '')} | {row.get('test_f1_macro', '')} "
                f"| {row.get('test_balanced_accuracy', '')} |\n"
            )
    else:
        handle.write("No runs were recorded.\n")

with calibration_index.open("w", encoding="utf-8") as handle:
    handle.write("# Calibration Plot Index\n\n")
    for row in summary_rows:
        plot_path = row.get("calibration_plot", "")
        if plot_path:
            handle.write(f"- `{row['experiment_name']}`: `{plot_path}`\n")
PY

echo
echo "Ablation summary CSV: $SUMMARY_CSV"
echo "Ablation summary MD:  $SUMMARY_MD"
echo "Calibration index:    $CALIBRATION_INDEX"
echo "Calibration plots:    $CALIBRATION_DIR"

if [[ ${#failures[@]} -gt 0 ]]; then
  echo
  echo "Failed experiments:"
  for name in "${failures[@]}"; do
    echo "- $name"
  done
  exit 1
fi

echo
echo "All ablation experiments completed successfully."
