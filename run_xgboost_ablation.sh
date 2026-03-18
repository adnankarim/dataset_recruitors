#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FEATURE_DIR="${FEATURE_DIR:-$ROOT_DIR/output/feature_splits}"
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

ALL_COSINE="pplx_qc_cosine_256,pplx_uc_cosine_256,pplx_qu_cosine_256,pplx_quc_cosine_256,pplx_qc_cosine_512,pplx_uc_cosine_512,pplx_qu_cosine_512,pplx_quc_cosine_512,pplx_qc_cosine_1024,pplx_uc_cosine_1024,pplx_qu_cosine_1024,pplx_quc_cosine_1024"
DIM_256="pplx_qc_cosine_256,pplx_uc_cosine_256,pplx_qu_cosine_256,pplx_quc_cosine_256"
DIM_512="pplx_qc_cosine_512,pplx_uc_cosine_512,pplx_qu_cosine_512,pplx_quc_cosine_512"
DIM_1024="pplx_qc_cosine_1024,pplx_uc_cosine_1024,pplx_qu_cosine_1024,pplx_quc_cosine_1024"
QC_ONLY="pplx_qc_cosine_256,pplx_qc_cosine_512,pplx_qc_cosine_1024"
UC_ONLY="pplx_uc_cosine_256,pplx_uc_cosine_512,pplx_uc_cosine_1024"
QU_ONLY="pplx_qu_cosine_256,pplx_qu_cosine_512,pplx_qu_cosine_1024"
QUC_ONLY="pplx_quc_cosine_256,pplx_quc_cosine_512,pplx_quc_cosine_1024"
QC_UC="pplx_qc_cosine_256,pplx_uc_cosine_256,pplx_qc_cosine_512,pplx_uc_cosine_512,pplx_qc_cosine_1024,pplx_uc_cosine_1024"
QC_QUC="pplx_qc_cosine_256,pplx_quc_cosine_256,pplx_qc_cosine_512,pplx_quc_cosine_512,pplx_qc_cosine_1024,pplx_quc_cosine_1024"

mkdir -p "$RUNS_DIR" "$LOGS_DIR" "$REPORTS_DIR" "$CALIBRATION_DIR"

MANIFEST_PATH="$REPORTS_DIR/ablation_manifest.tsv"
SUMMARY_CSV="$REPORTS_DIR/ablation_summary.csv"
SUMMARY_MD="$REPORTS_DIR/ablation_summary.md"
CALIBRATION_INDEX="$REPORTS_DIR/calibration_index.md"

cat >"$MANIFEST_PATH" <<EOF
experiment_name	class_imbalance_handling	feature_columns	run_dir	log_path	status
EOF

readarray -t EXPERIMENT_SPECS <<EOF
all_cosine_off	off	$ALL_COSINE
all_cosine_balanced	balanced-sample-weight	$ALL_COSINE
dim256_balanced	balanced-sample-weight	$DIM_256
dim512_balanced	balanced-sample-weight	$DIM_512
dim1024_balanced	balanced-sample-weight	$DIM_1024
qc_only_balanced	balanced-sample-weight	$QC_ONLY
uc_only_balanced	balanced-sample-weight	$UC_ONLY
qu_only_balanced	balanced-sample-weight	$QU_ONLY
quc_only_balanced	balanced-sample-weight	$QUC_ONLY
qc_uc_balanced	balanced-sample-weight	$QC_UC
qc_quc_balanced	balanced-sample-weight	$QC_QUC
EOF

echo "Ablation root: $ABLATION_DIR"
echo "Using feature dir: $FEATURE_DIR"
echo "Using python: $PYTHON_BIN"

failures=()

run_experiment() {
  local name="$1"
  local imbalance="$2"
  local features="$3"
  local run_dir="$RUNS_DIR/$name"
  local log_path="$LOGS_DIR/$name.log"
  local calibration_target="$CALIBRATION_DIR/${name}_calibration.png"
  local status="success"

  mkdir -p "$run_dir"

  echo
  echo "============================================================"
  echo "Running experiment: $name"
  echo "Class imbalance:   $imbalance"
  echo "Features:          $features"
  echo "Run dir:           $run_dir"
  echo "============================================================"

  if "$PYTHON_BIN" "$ROOT_DIR/train_xgboost_on_pplx_features.py" \
    --feature-dir "$FEATURE_DIR" \
    --run-dir "$run_dir" \
    --feature-cols "$features" \
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
    "$name" "$imbalance" "$features" "$run_dir" "$log_path" "$status" >>"$MANIFEST_PATH"
}

for spec in "${EXPERIMENT_SPECS[@]}"; do
  IFS=$'\t' read -r name imbalance features <<<"$spec"
  run_experiment "$name" "$imbalance" "$features"
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
    for row in reader:
        rows.append(row)

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
        "feature_columns": row["feature_columns"],
        "run_dir": row["run_dir"],
        "log_path": row["log_path"],
        "calibration_plot": str(calibration_path) if calibration_path.exists() else "",
    }

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for split_name in ("valid", "test", "train"):
            split_metrics = metrics.get(split_name, {})
            for key, value in split_metrics.items():
                record[f"{split_name}_{key}"] = value

    if summary_path.exists():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record["class_names"] = ",".join(run_summary.get("class_names", []))
        record["feature_count"] = run_summary.get("feature_count")
        record["class_weight_map"] = json.dumps(run_summary.get("class_weight_map"), ensure_ascii=True)

    summary_rows.append(record)

sort_keys = [
    "valid_f1_macro",
    "valid_balanced_accuracy",
    "valid_accuracy",
]

def sort_value(row, key):
    value = row.get(key)
    if value in (None, "", "None"):
        return float("-inf")
    return float(value)

summary_rows.sort(key=lambda row: tuple(sort_value(row, key) for key in sort_keys), reverse=True)

fieldnames = [
    "experiment_name",
    "status",
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
    "class_weight_map",
    "feature_columns",
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
    handle.write("# XGBoost Ablation Summary\n\n")
    if summary_rows:
        best = summary_rows[0]
        handle.write("## Best Validation Run\n\n")
        handle.write(f"- Experiment: `{best['experiment_name']}`\n")
        handle.write(f"- Class imbalance handling: `{best['class_imbalance_handling']}`\n")
        handle.write(f"- Validation macro F1: `{best.get('valid_f1_macro', '')}`\n")
        handle.write(f"- Validation balanced accuracy: `{best.get('valid_balanced_accuracy', '')}`\n")
        handle.write(f"- Test macro F1: `{best.get('test_f1_macro', '')}`\n")
        handle.write(f"- Run directory: `{best['run_dir']}`\n\n")

        handle.write("## All Runs\n\n")
        handle.write("| Experiment | Status | Imbalance | Valid F1 Macro | Valid Balanced Acc | Test F1 Macro | Test Balanced Acc |\n")
        handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: |\n")
        for row in summary_rows:
            handle.write(
                f"| {row['experiment_name']} | {row['status']} | {row['class_imbalance_handling']} "
                f"| {row.get('valid_f1_macro', '')} | {row.get('valid_balanced_accuracy', '')} "
                f"| {row.get('test_f1_macro', '')} | {row.get('test_balanced_accuracy', '')} |\n"
            )
        handle.write("\n")
        handle.write("## Files\n\n")
        handle.write(f"- CSV summary: `{summary_csv}`\n")
        handle.write(f"- Calibration plot index: `{calibration_index}`\n")
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
