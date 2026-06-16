#!/usr/bin/env bash
# Merge per-question JSONs from s3_llm_final_ans and run LLM-based accuracy eval.
#
# WORKDIR is read from the environment (set/exported by TimeProVe_*.sh) or passed
# as the first argument:
#   bash merge_jsons_n_eval_otb.sh
#   bash merge_jsons_n_eval_otb.sh /path/to/workdir
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation"

if [[ -n "${1:-}" ]]; then
  WORKDIR="$1"
elif [[ -z "${WORKDIR:-}" ]]; then
  echo "ERROR: WORKDIR is not set. Run via a TimeProVe pipeline script or pass a workdir:" >&2
  echo "  WORKDIR=/path/to/workdir bash $0" >&2
  echo "  bash $0 /path/to/workdir" >&2
  exit 1
fi

cd "$EVAL_DIR"
python eval_merge_topk_output_jsons.py \
  -i "$WORKDIR/s3_llm_final_ans" \
  -o "$WORKDIR/merged.json"

python evaluate_accuracy_llm_parse.py \
  -i "$WORKDIR/merged.json" \
  -o "$WORKDIR/final_results.json"
