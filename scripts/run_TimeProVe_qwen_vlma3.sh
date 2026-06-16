#!/usr/bin/env bash
# Run the interpretable VLM pipeline on OTB using MS-TEMBA AD predictions.
# (pkl predictions -> threshold -> segments -> descriptions -> final QA)
#
# Usage:
#   ./run_mstemba_interpretable_llm_newparser_pipeline_sample_OTB.sh
#   SAMPLE_COUNT=100 GPU_IDS=0,1 AD_THRESHOLD=0.1 \
#     ./run_mstemba_interpretable_llm_newparser_pipeline_sample_OTB.sh --resume
#
# Key env-var overrides:
#   MODEL_PATH         HF id or local checkpoint (default: project fine-tune)
#   GPU_IDS            Comma-separated GPU indices to use (default: 0,1,2,3)
#   SAMPLE_COUNT       Max questions to prepare (default: 100; 0 = all)
#   AD_THRESHOLD       Raw probability threshold on pkl scores (no sigmoid; default: 0.50)
#   POOL_FACTOR        Temporal pooling factor (pkl_frame × pool_factor = video frame; default: 16)
#   MERGE_GAP_FRAMES   Max pooled-frame gap to bridge when merging same-class segments (default: 1)
#   EXTRACT_CONTEXT_SEC  Temporal padding added around each clip (default: 7.0)
#   EXTRACT_MIN_CLIP_SEC Minimum clip length in seconds (default: 15.0)
#   BENCHMARK_JSON     Path to stage3_quality_checked_final.json
#   PKL_PATH           Path to AD pkl (default: data/TSU_best_AD.pkl)
#   OTB_ACTION_LIST    Path to OTB action-list file (default: data/TSU_Action_list.txt)
#   OTB_VIDEO_ROOT Directory containing <video_id>.mp4 files
#   DATA_PATH          Prepared questions JSON (written by this script if absent)
#   WORKDIR            Root workdir (overrides FINAL_OUT_DIR / S2_OUT_DIR defaults)
#   FINAL_OUT_DIR      Per-question final-answer JSON output directory
#   S2_OUT_DIR         Segment descriptions + llm_actions output directory
#   VIDEO_SEG_ROOT     Extracted video clip root (default: $S2_OUT_DIR/video_segments)
#   LOG_DIR            Directory for per-GPU log files
#
# λ grid search (use a distinct WORKDIR per combo so outputs do not overwrite):
#   for SCORE_LAM_SEMANTIC in 1.0 0.5; do
#     for SCORE_LAM_TEMPORAL in 0.4 0.2; do
#       export SCORE_LAM_SEMANTIC SCORE_LAM_TEMPORAL
#       export WORKDIR="$REPO_ROOT/workdirs/OTB/ablations/lam1_${SCORE_LAM_SEMANTIC}_lam2_${SCORE_LAM_TEMPORAL}"
#       ./run_final_pipeline_MStemba_qwen_vlma3_OTB_ablations_28Apr26.sh
#     done
#   done
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation"
cd "$EVAL_DIR"

# ── Configurable defaults ──────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/home/asinha13/projects/LLM_Token_Selector/VideoLLaMA3/weights/videollama3_7b_local}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SAMPLE_COUNT="${SAMPLE_COUNT:-0}"
AD_THRESHOLD="${AD_THRESHOLD:-0.50}"
POOL_FACTOR="${POOL_FACTOR:-16}"
MERGE_GAP_FRAMES="${MERGE_GAP_FRAMES:-1}"
EXTRACT_CONTEXT_SEC="${EXTRACT_CONTEXT_SEC:-7.0}"
EXTRACT_MIN_CLIP_SEC="${EXTRACT_MIN_CLIP_SEC:-15.0}"
# Scoring ablation weights (pipeline v29Apr26):
#   lam1=semantic, lam2=temporal, lam3=coverage, lam4=cost, lam5=feedback, lam6=confidence
# Default here sets lam1=0 for semantic-ablation experiments.
SCORE_LAM_SEMANTIC="${SCORE_LAM_SEMANTIC:-1.0}"
SCORE_LAM_TEMPORAL="${SCORE_LAM_TEMPORAL:-1.0}"
SCORE_LAM_COVERAGE="${SCORE_LAM_COVERAGE:-1.0}"
SCORE_LAM_COST="${SCORE_LAM_COST:-1.0}"
SCORE_LAM_FEEDBACK="${SCORE_LAM_FEEDBACK:-1.0}"
SCORE_LAM_CONFIDENCE="${SCORE_LAM_CONFIDENCE:-0.00}"

BENCHMARK_JSON="${BENCHMARK_JSON:-$REPO_ROOT/data/otb_bench.json}"
PKL_PATH="${PKL_PATH:-$REPO_ROOT/data/TSU_best_AD.pkl}"
OTB_ACTION_LIST="${OTB_ACTION_LIST:-$REPO_ROOT/data/TSU_Action_list.txt}"
OTB_VIDEO_ROOT="${OTB_VIDEO_ROOT:-/data/vidlab_datasets/smarthome/untrimmed/Videos_mp4}"

WORKDIR="${WORKDIR:-$REPO_ROOT/workdirs/OTB_mstemba_qwen_draft_vlma3_target_15June26}"
export WORKDIR
PREP_DIR="${PREP_DIR:-$WORKDIR/prepared_inputs}"
DATA_PATH="${DATA_PATH:-$PREP_DIR/otb_mstemba_samples.json}"
FINAL_OUT_DIR="${FINAL_OUT_DIR:-$WORKDIR/s3_llm_final_ans}"
S2_OUT_DIR="${S2_OUT_DIR:-$WORKDIR/s2_vlm_desc}"
VIDEO_SEG_ROOT="${VIDEO_SEG_ROOT:-$S2_OUT_DIR/video_segments}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/workdirs/logs_otb_mstemba}"

mkdir -p "$PREP_DIR" "$FINAL_OUT_DIR" "$LOG_DIR"

# ── Step 1: Prepare flat question JSON ────────────────────────────────────────
# Selects up to SAMPLE_COUNT questions from the benchmark, keeping only videos
# that exist in both the pkl and on disk.  Writes DATA_PATH if it does not yet
# exist (or if it is empty), so --resume runs can skip this step.

if [[ -f "$DATA_PATH" ]] && [[ "$(python - "$DATA_PATH" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text()) if p.stat().st_size > 0 else {}
print(len(d) if isinstance(d, dict) else 0)
PY
)" -gt 0 ]]; then
  echo "Data file already exists and is non-empty, skipping preparation: $DATA_PATH"
else
  echo "Preparing question data → $DATA_PATH"
  python - \
    "$BENCHMARK_JSON" \
    "$PKL_PATH" \
    "$OTB_VIDEO_ROOT" \
    "$SAMPLE_COUNT" \
    "$DATA_PATH" <<'PY'
import json
import pickle
import sys
from pathlib import Path

benchmark_path = Path(sys.argv[1])
pkl_path       = Path(sys.argv[2])
video_root     = Path(sys.argv[3])
sample_count   = int(sys.argv[4])
out_data_path  = Path(sys.argv[5])

if not benchmark_path.is_file():
    raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
if not pkl_path.is_file():
    raise FileNotFoundError(f"Pkl file not found: {pkl_path}")

benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
if not isinstance(benchmark, dict):
    raise TypeError(f"Expected dict in benchmark json, got {type(benchmark).__name__}")

with open(pkl_path, "rb") as fh:
    pkl_data = pickle.load(fh)
if not isinstance(pkl_data, dict):
    raise TypeError(f"Expected dict in pkl, got {type(pkl_data).__name__}")

pkl_video_ids = set(pkl_data.keys())
print(f"pkl contains {len(pkl_video_ids)} video(s)")


def resolve_video_path(root: Path, video_id: str) -> Path | None:
    for p in [
        root / f"{video_id}.mp4",
        root / f"{video_id}.MP4",
        root / video_id / f"{video_id}.mp4",
        root / video_id / f"{video_id}.MP4",
    ]:
        if p.is_file():
            return p
    for ext in ("mp4", "MP4"):
        for p in root.rglob(f"{video_id}.{ext}"):
            if p.is_file():
                return p
    return None


selected: dict[str, dict] = {}
skipped_no_pkl = 0
skipped_no_video = 0

for video_id, video_rec in benchmark.items():
    if video_id not in pkl_video_ids:
        skipped_no_pkl += 1
        continue
    if resolve_video_path(video_root, video_id) is None:
        skipped_no_video += 1
        continue
    if not isinstance(video_rec, dict):
        continue
    candidates = video_rec.get("candidates")
    if not isinstance(candidates, list):
        continue
    for idx, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            continue
        question = (cand.get("question") or cand.get("canonical_question") or "").strip()
        if not question:
            continue
        qid = str(cand.get("candidate_id") or f"{video_id}_cand{idx:04d}")
        if qid in selected:
            suffix = 2
            while f"{qid}__{suffix}" in selected:
                suffix += 1
            qid = f"{qid}__{suffix}"
        selected[qid] = {
            "question": question,
            "answer": cand.get("answer"),
            "video_id": video_id,
        }
        if sample_count > 0 and len(selected) >= sample_count:
            break
    if sample_count > 0 and len(selected) >= sample_count:
        break

if not selected:
    raise RuntimeError(
        "No valid OTB samples found. "
        f"skipped_no_pkl={skipped_no_pkl} skipped_no_video={skipped_no_video}"
    )

if sample_count > 0 and len(selected) < sample_count:
    print(
        f"Warning: requested {sample_count} sample(s), found {len(selected)} valid sample(s).",
        file=sys.stderr,
    )

out_data_path.parent.mkdir(parents=True, exist_ok=True)
out_data_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(
    f"Prepared {len(selected)} sample(s). "
    f"skipped_no_pkl={skipped_no_pkl} skipped_no_video={skipped_no_video}. "
    f"data={out_data_path}"
)
PY
fi

# ── Step 2: Count questions for the progress monitor ─────────────────────────
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_SHARDS="${#GPU_ARRAY[@]}"

if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "No GPUs found in GPU_IDS='$GPU_IDS'" >&2
  exit 1
fi

START_COUNT="$(ls -1 "$FINAL_OUT_DIR"/*.json 2>/dev/null | wc -l || true)"
TARGET_TOTAL="$(
python - "$DATA_PATH" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print(0); raise SystemExit(0)
raw = p.read_text(encoding="utf-8").strip()
d = json.loads(raw) if raw else {}
print(len(d) if isinstance(d, dict) else 0)
PY
)"

echo ""
echo "Launching ${NUM_SHARDS} shard(s) on GPUs: ${GPU_IDS}"
echo "  BENCHMARK_JSON     = $BENCHMARK_JSON"
echo "  PKL_PATH           = $PKL_PATH"
echo "  OTB_ACTION_LIST    = $OTB_ACTION_LIST"
echo "  OTB_VIDEO_ROOT= $OTB_VIDEO_ROOT"
echo "  DATA_PATH          = $DATA_PATH"
echo "  AD_THRESHOLD       = $AD_THRESHOLD"
echo "  POOL_FACTOR        = $POOL_FACTOR"
echo "  MERGE_GAP_FRAMES   = $MERGE_GAP_FRAMES"
echo "  EXTRACT_CONTEXT_SEC= $EXTRACT_CONTEXT_SEC"
echo "  EXTRACT_MIN_CLIP_SEC=$EXTRACT_MIN_CLIP_SEC"
echo "  SCORE_LAM_SEMANTIC = $SCORE_LAM_SEMANTIC"
echo "  SCORE_LAM_TEMPORAL = $SCORE_LAM_TEMPORAL"
echo "  SCORE_LAM_COVERAGE = $SCORE_LAM_COVERAGE"
echo "  SCORE_LAM_COST     = $SCORE_LAM_COST"
echo "  SCORE_LAM_FEEDBACK = $SCORE_LAM_FEEDBACK"
echo "  SCORE_LAM_CONFIDENCE=$SCORE_LAM_CONFIDENCE"
echo "  FINAL_OUT_DIR      = $FINAL_OUT_DIR"
echo "  S2_OUT_DIR         = $S2_OUT_DIR"
echo "  VIDEO_SEG_ROOT     = $VIDEO_SEG_ROOT"
echo "  LOG_DIR            = $LOG_DIR"
echo "Progress target: ${TARGET_TOTAL} total question(s), ${START_COUNT} already completed"

# ── Step 3: Launch one shard per GPU ─────────────────────────────────────────
pids=()
for shard_idx in "${!GPU_ARRAY[@]}"; do
  gpu_id="${GPU_ARRAY[$shard_idx]}"
  log_file="$LOG_DIR/run_mstemba_pipeline_otb_gpu${gpu_id}.log"
  echo "Starting shard ${shard_idx}/${NUM_SHARDS} on GPU ${gpu_id} → $log_file"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    python TimeProVe_qwen_vlma3.py \
      --data              "$DATA_PATH" \
      --classes-file      "$OTB_ACTION_LIST" \
      --pkl               "$PKL_PATH" \
      --extract-threshold "$AD_THRESHOLD" \
      --pool-factor       "$POOL_FACTOR" \
      --merge-gap-frames  "$MERGE_GAP_FRAMES" \
      --otb-video-root "$OTB_VIDEO_ROOT" \
      --extract-context-sec "$EXTRACT_CONTEXT_SEC" \
      --extract-min-clip-sec "$EXTRACT_MIN_CLIP_SEC" \
      --score-lam-semantic "$SCORE_LAM_SEMANTIC" \
      --score-lam-temporal "$SCORE_LAM_TEMPORAL" \
      --score-lam-coverage "$SCORE_LAM_COVERAGE" \
      --score-lam-cost     "$SCORE_LAM_COST" \
      --score-lam-feedback "$SCORE_LAM_FEEDBACK" \
      --score-lam-confidence "$SCORE_LAM_CONFIDENCE" \
      --model-path        "$MODEL_PATH" \
      --device            "cuda:0" \
      --num-shards        "$NUM_SHARDS" \
      --shard-index       "$shard_idx" \
      --final-out-dir     "$FINAL_OUT_DIR" \
      --s2-out            "$S2_OUT_DIR" \
      --video-seg-root    "$VIDEO_SEG_ROOT" \
      "$@" >"$log_file" 2>&1 &
  pids+=("$!")
done

# ── Step 4: Progress monitor ──────────────────────────────────────────────────
monitor_pid=""
if [[ "$TARGET_TOTAL" -gt 0 ]]; then
  python - "$FINAL_OUT_DIR" "$TARGET_TOTAL" "$START_COUNT" "${pids[@]}" <<'PY' &
import glob, os, sys, time
from pathlib import Path

out_dir  = Path(sys.argv[1])
total    = int(sys.argv[2])
initial  = int(sys.argv[3])
pids     = [int(x) for x in sys.argv[4:]]

def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def count_done() -> int:
    return len(glob.glob(str(out_dir / "*.json")))

try:
    from tqdm import tqdm
    bar = tqdm(total=total, initial=min(initial, total),
               desc="Overall progress", dynamic_ncols=True)
    while any(running(p) for p in pids):
        bar.n = min(count_done(), total)
        bar.refresh()
        time.sleep(2.0)
    bar.n = min(count_done(), total)
    bar.refresh()
    bar.close()
except Exception:
    while any(running(p) for p in pids):
        done = min(count_done(), total)
        print(f"[overall] {done}/{total}", flush=True)
        time.sleep(5.0)
    done = min(count_done(), total)
    print(f"[overall] {done}/{total}", flush=True)
PY
  monitor_pid="$!"
fi

# ── Step 5: Wait and report ───────────────────────────────────────────────────
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ -n "$monitor_pid" ]]; then
  wait "$monitor_pid" || true
fi

if [[ "$failed" -ne 0 ]]; then
  echo "One or more shard processes failed." >&2
  exit 1
fi

echo "All shards completed successfully."

python pipeline_cleanup.py \
  --data "$DATA_PATH" \
  --final-out-dir "$FINAL_OUT_DIR" \
  --s2-out "$S2_OUT_DIR" \
  -v

WORKDIR="$WORKDIR" bash "$SCRIPT_DIR/merge_jsons_n_eval_otb.sh"