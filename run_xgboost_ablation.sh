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
DEVICE="${DEVICE:-cpu}"
NUM_BOOST_ROUND="${NUM_BOOST_ROUND:-1200}"
EARLY_STOPPING_ROUNDS="${EARLY_STOPPING_ROUNDS:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
SEED="${SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DIMS="256,512,1024"
IMBALANCE_MODES="off,balanced-sample-weight"

usage() {
  cat <<EOF
Usage: bash run_xgboost_ablation.sh [--dims 256,512,1024] [--imbalance-modes off,balanced-sample-weight]

Options:
  --dims               Comma-separated embedding prefix dimensions to evaluate. Default: 256,512,1024
  --imbalance-modes    Comma-separated imbalance modes. Default: off,balanced-sample-weight
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dims)
      DIMS="$2"
      shift 2
      ;;
    --imbalance-modes)
      IMBALANCE_MODES="$2"
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

IFS=',' read -r -a DIM_VALUES <<<"$DIMS"
IFS=',' read -r -a IMBALANCE_VALUES <<<"$IMBALANCE_MODES"

mkdir -p "$RUNS_DIR" "$LOGS_DIR" "$REPORTS_DIR" "$CALIBRATION_DIR"

MANIFEST_PATH="$REPORTS_DIR/ablation_manifest.tsv"
SUMMARY_CSV="$REPORTS_DIR/ablation_summary.csv"
SUMMARY_MD="$REPORTS_DIR/ablation_summary.md"
CALIBRATION_INDEX="$REPORTS_DIR/calibration_index.md"

cat >"$MANIFEST_PATH" <<EOF
experiment_name	class_imbalance_handling	embedding_prefix_dim	run_dir	log_path	status
EOF

echo "Ablation root: $ABLATION_DIR"
echo "Using feature dir: $FEATURE_DIR"
echo "Using store dir:   $STORE_DIR"
echo "Using python: $PYTHON_BIN"
echo "Embedding dims: $DIMS"
echo "Imbalance modes: $IMBALANCE_MODES"

failures=()

run_experiment() {
  local name="$1"
  local imbalance="$2"
  local prefix_dim="$3"
  local run_dir="$RUNS_DIR/$name"
  local log_path="$LOGS_DIR/$name.log"
  local calibration_target="$CALIBRATION_DIR/${name}_calibration.png"
  local status="success"

  mkdir -p "$run_dir"

  echo
  echo "============================================================"
  echo "Running experiment: $name"
  echo "Class imbalance:   $imbalance"
  echo "Embedding dim:     $prefix_dim"
  echo "Run dir:           $run_dir"
  echo "============================================================"

  if "$PYTHON_BIN" "$ROOT_DIR/train_xgboost_on_pplx_features.py" \
    --feature-dir "$FEATURE_DIR" \
    --store-dir "$STORE_DIR" \
    --run-dir "$run_dir" \
    --embedding-prefix-dim "$prefix_dim" \
    --class-imbalance-handling "$imbalance" \
    --device "$DEVICE" \
    --num-boost-round "$NUM_BOOST_ROUND" \
    --early-stopping-rounds "$EARLY_STOPPING_ROUNDS" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --seed "$SEED" \
    ${EXTRA_ARGS} 2>&1 | tee "$log_path"; then
    if [[ -f "$run_dir/plots/calibration_curves.png" ]]; then
      cp "$run_dir/plots/calibration_curves.png" "$calibration_target"
    fi
  else
    status="failed"
    failures+=("$name")
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$imbalance" "$prefix_dim" "$run_dir" "$log_path" "$status" >>"$MANIFEST_PATH"
}

for dim in "${DIM_VALUES[@]}"; do
  dim="$(echo "$dim" | xargs)"
  [[ -n "$dim" ]] || continue
  for imbalance in "${IMBALANCE_VALUES[@]}"; do
    imbalance="$(echo "$imbalance" | xargs)"
    [[ -n "$imbalance" ]] || continue
    run_experiment "rawdim${dim}_${imbalance//-/_}" "$imbalance" "$dim"
  done
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
    "embedding_prefix_dim",
    "class_imbalance_handling",
    "feature_count",
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
    handle.write("# Raw Embedding XGBoost Ablation Summary\n\n")
    if summary_rows:
        best = summary_rows[0]
        handle.write("## Best Validation Run\n\n")
        handle.write(f"- Experiment: `{best['experiment_name']}`\n")
        handle.write(f"- Embedding prefix dim: `{best['embedding_prefix_dim']}`\n")
        handle.write(f"- Class imbalance handling: `{best['class_imbalance_handling']}`\n")
        handle.write(f"- Validation macro F1: `{best.get('valid_f1_macro', '')}`\n")
        handle.write(f"- Validation balanced accuracy: `{best.get('valid_balanced_accuracy', '')}`\n")
        handle.write(f"- Test macro F1: `{best.get('test_f1_macro', '')}`\n")
        handle.write(f"- Run directory: `{best['run_dir']}`\n\n")
        handle.write("| Experiment | Dim | Imbalance | Valid F1 Macro | Valid Balanced Acc | Test F1 Macro | Test Balanced Acc |\n")
        handle.write("| --- | ---: | --- | ---: | ---: | ---: | ---: |\n")
        for row in summary_rows:
            handle.write(
                f"| {row['experiment_name']} | {row['embedding_prefix_dim']} | {row['class_imbalance_handling']} "
                f"| {row.get('valid_f1_macro', '')} | {row.get('valid_balanced_accuracy', '')} "
                f"| {row.get('test_f1_macro', '')} | {row.get('test_balanced_accuracy', '')} |\n"
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
