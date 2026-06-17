#!/usr/bin/env bash
# Run the GPT+MSTemba interpretable pipeline on OTB.
# MSTemba pkl predictions (thresholded) supply action names and temporal windows.
# GPT-4o (or another vision model) generates clip descriptions; Gemma4
# handles the text-only confidence check and final QA.
#
# Usage:
#   OPENAI_API_KEY=sk-... ./run_GPT_mstemba_Gemma_pipeline_OTB.sh
#   SAMPLE_COUNT=100 GPU_IDS=0,1 GPT_VLM_MODEL=gpt-4o-mini ./run_GPT_mstemba_Gemma_pipeline_OTB.sh --resume
#   SCORE_LAM_SEMANTIC=1.0 SCORE_LAM_TEMPORAL=0.4 ... ./run_GPT_mstemba_Gemma_pipeline_OTB.sh
#
# Gemma GPU layout (matches pipeline --gemma4-gpu-ids / --gemma4-device-map):
#   - Multi-shard (default): GPU_IDS lists one physical GPU per process. Each shard sets
#     CUDA_VISIBLE_DEVICES to that GPU; use GEMMA4_DEVICE_MAP=cuda:0 (default). Do not set
#     GEMMA4_GPU_IDS (each process only sees one GPU).
#   - Single-process multi-GPU Gemma: set GPU_IDS to exactly one entry (shard count = 1),
#     set GEMMA4_GPU_IDS="2,3,4" (physical IDs), and GEMMA4_DEVICE_MAP=auto
#     (needs pip install accelerate). Only one Python job will run in that case.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation"
cd "$EVAL_DIR"

# ── Gemma4 (text-only LLM for confidence + final QA) ───────────────────────
MODEL_PATH="${MODEL_PATH:-google/gemma-4-E2B-it}"
# Comma-separated physical GPU IDs for Gemma only when running a single pipeline process
# (see header). Leave empty for normal multi-shard runs (each shard uses one GPU via CUDA_VISIBLE_DEVICES).
GEMMA4_GPU_IDS="${GEMMA4_GPU_IDS:-}"
GEMMA4_DEVICE_MAP="${GEMMA4_DEVICE_MAP:-cuda:0}"
GEMMA4_DTYPE="${GEMMA4_DTYPE:-float16}"
# Sharding: one parallel Python process per listed GPU (data parallel over questions).
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

# ── OpenAI / GPT VLM settings ──────────────────────────────────────────────
OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_API_KEY}"
GPT_VLM_MODEL="${GPT_VLM_MODEL:-gpt-4o}"
GPT_DESC_MAX_FRAMES="${GPT_DESC_MAX_FRAMES:-20}"

if [[ -z "$OPENAI_API_KEY" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Export it or prefix the command:" >&2
  echo "  OPENAI_API_KEY=sk-... $0" >&2
  exit 1
fi

# ── MSTemba pkl settings ────────────────────────────────────────────────────
PKL_PATH="${PKL_PATH:-$REPO_ROOT/data/TSU_best_AD.pkl}"
EXTRACT_THRESHOLD="${EXTRACT_THRESHOLD:-0.50}"
POOL_FACTOR="${POOL_FACTOR:-16}"
MERGE_GAP_FRAMES="${MERGE_GAP_FRAMES:-1}"

# Scoring weights (λ1…λ6): semantic, temporal, coverage, cost, feedback, detection confidence
SCORE_LAM_SEMANTIC="${SCORE_LAM_SEMANTIC:-1.0}"
SCORE_LAM_TEMPORAL="${SCORE_LAM_TEMPORAL:-1.0}"
SCORE_LAM_COVERAGE="${SCORE_LAM_COVERAGE:-1.0}"
SCORE_LAM_COST="${SCORE_LAM_COST:-1.0}"
SCORE_LAM_FEEDBACK="${SCORE_LAM_FEEDBACK:-1.0}"
SCORE_LAM_CONFIDENCE="${SCORE_LAM_CONFIDENCE:-0.00}"

# ── Dataset / sampling ─────────────────────────────────────────────────────
SAMPLE_COUNT="${SAMPLE_COUNT:-3567}"
EXTRACT_CONTEXT_SEC="${EXTRACT_CONTEXT_SEC:-7.0}"
EXTRACT_MIN_CLIP_SEC="${EXTRACT_MIN_CLIP_SEC:-15.0}"

BENCHMARK_JSON="${BENCHMARK_JSON:-$REPO_ROOT/data/otb_bench.json}"
OTB_ANNOTATIONS="${OTB_ANNOTATIONS:-$REPO_ROOT/data/smarthome.json}"
OTB_ACTION_LIST="${OTB_ACTION_LIST:-$REPO_ROOT/data/TSU_Action_list.txt}"
OTB_VIDEO_ROOT="${OTB_VIDEO_ROOT:-/data/vidlab_datasets/smarthome/untrimmed/Videos_mp4}"
PREP_DIR="${PREP_DIR:-$REPO_ROOT/workdirs/otb_all_samples_gtquery_actnprop_desc/prepared_inputs}"
DATA_PATH="${DATA_PATH:-$PREP_DIR/otb_all_samples.json}"

FINAL_OUT_DIR="${FINAL_OUT_DIR:-$REPO_ROOT/workdirs/OTB/ablations/OTB_mstemba_GPT_Gemma_20May26/s3_llm_final_ans}"
S2_OUT_DIR="${S2_OUT_DIR:-$REPO_ROOT/workdirs/OTB/ablations/OTB_mstemba_GPT_Gemma_20May26/s2_vlm_desc}"
VIDEO_SEG_ROOT="${VIDEO_SEG_ROOT:-$S2_OUT_DIR/video_segments}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs_otb_gpt_mstemba}"

mkdir -p "$PREP_DIR" "$FINAL_OUT_DIR" "$LOG_DIR"

# ── Prepare OTB question data (MSTemba pipeline uses pkl, not GT annotations) ─
python - "$BENCHMARK_JSON" "$OTB_ANNOTATIONS" "$OTB_VIDEO_ROOT" "$SAMPLE_COUNT" "$DATA_PATH" <<'PY'
import json
import math
import re
import subprocess
import sys
from pathlib import Path

benchmark_path = Path(sys.argv[1])
otb_path = Path(sys.argv[2])
video_root = Path(sys.argv[3])
sample_count = int(sys.argv[4])
out_data_path = Path(sys.argv[5])

if sample_count <= 0:
    raise ValueError(f"SAMPLE_COUNT must be >= 1, got {sample_count}")
if not benchmark_path.is_file():
    raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
if not otb_path.is_file():
    raise FileNotFoundError(f"OTB annotations not found: {otb_path}")

benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
annotations = json.loads(otb_path.read_text(encoding="utf-8"))
if not isinstance(benchmark, dict):
    raise TypeError(f"Expected dict in benchmark json, got {type(benchmark).__name__}")
if not isinstance(annotations, dict):
    raise TypeError(f"Expected dict in otb annotations, got {type(annotations).__name__}")

def ffprobe_video_meta(video_path: Path) -> tuple:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate:format=duration",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(out.stdout or "{}")
        duration = None
        fmt = payload.get("format")
        if isinstance(fmt, dict):
            d = fmt.get("duration")
            if isinstance(d, str) and d.strip():
                duration = float(d)
        fps = None
        streams = payload.get("streams")
        if isinstance(streams, list) and streams:
            s0 = streams[0] if isinstance(streams[0], dict) else {}
            rf = s0.get("r_frame_rate")
            if isinstance(rf, str) and "/" in rf:
                num_s, den_s = rf.split("/", 1)
                num, den = float(num_s), float(den_s)
                if den != 0:
                    fps = num / den
        return duration, fps
    except Exception:
        return None, None


def resolve_video_path(root: Path, video_id: str):
    direct_candidates = [
        root / f"{video_id}.mp4",
        root / f"{video_id}.MP4",
        root / video_id / f"{video_id}.mp4",
        root / video_id / f"{video_id}.MP4",
    ]
    for p in direct_candidates:
        if p.is_file():
            return p
    for p in root.rglob(f"{video_id}.mp4"):
        if p.is_file():
            return p
    for p in root.rglob(f"{video_id}.MP4"):
        if p.is_file():
            return p
    return None


video_path_by_id = {}
for video_id in benchmark.keys():
    p = resolve_video_path(video_root, video_id)
    if p is not None:
        video_path_by_id[video_id] = p

selected = {}
selected_video_ids = []
for video_id, video_rec in benchmark.items():
    if video_id not in annotations:
        continue
    if video_id not in video_path_by_id:
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
        if video_id not in selected_video_ids:
            selected_video_ids.append(video_id)
        if len(selected) >= sample_count:
            break
    if len(selected) >= sample_count:
        break

if not selected:
    raise RuntimeError("No valid OTB samples were selected.")

if len(selected) < sample_count:
    print(
        f"Warning: requested {sample_count} sample(s), found {len(selected)} valid sample(s).",
        file=sys.stderr,
    )

out_data_path.parent.mkdir(parents=True, exist_ok=True)
out_data_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print(
    f"Prepared {len(selected)} sample(s). "
    f"resolved_videos={len(video_path_by_id)}. "
    f"data={out_data_path}"
)
PY

# ── Launch parallel shards ──────────────────────────────────────────────────
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_SHARDS="${#GPU_ARRAY[@]}"

if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "No GPUs found in GPU_IDS='$GPU_IDS'" >&2
  exit 1
fi

GEMMA_EXTRA_ARGS=()
if [[ -n "$GEMMA4_GPU_IDS" ]]; then
  if [[ "$NUM_SHARDS" -gt 1 ]]; then
    echo "ERROR: GEMMA4_GPU_IDS is set ($GEMMA4_GPU_IDS) but GPU_IDS defines ${NUM_SHARDS} shards." >&2
    echo "  For multi-shard runs, leave GEMMA4_GPU_IDS empty; each shard uses one GPU via CUDA_VISIBLE_DEVICES." >&2
    echo "  For single-process multi-GPU Gemma, set GPU_IDS to exactly one shard (e.g. GPU_IDS=0) and set GEMMA4_GPU_IDS." >&2
    exit 1
  fi
  GEMMA_EXTRA_ARGS+=(--gemma4-gpu-ids "$GEMMA4_GPU_IDS")
  # If user did not override, spread Gemma across the listed GPUs (requires accelerate).
  _gemma_id_count="${GEMMA4_GPU_IDS//[^,]/}"
  _gemma_id_count="${#_gemma_id_count}"
  _gemma_id_count=$((_gemma_id_count + 1))
  if [[ "$_gemma_id_count" -gt 1 && "$GEMMA4_DEVICE_MAP" == "cuda:0" ]]; then
    GEMMA4_DEVICE_MAP="auto"
    echo "INFO: GEMMA4_GPU_IDS lists ${_gemma_id_count} GPU(s); using GEMMA4_DEVICE_MAP=auto"
  fi
fi

echo "Launching ${NUM_SHARDS} parallel shard(s) on GPUs: ${GPU_IDS}"
echo "Gemma4: model=${MODEL_PATH}  GEMMA4_GPU_IDS=${GEMMA4_GPU_IDS:-"(unset)"}  GEMMA4_DEVICE_MAP=${GEMMA4_DEVICE_MAP}  GEMMA4_DTYPE=${GEMMA4_DTYPE}"
echo "GPT VLM model: ${GPT_VLM_MODEL}  max_frames_per_clip: ${GPT_DESC_MAX_FRAMES}"
echo "MSTemba pkl: ${PKL_PATH}  threshold: ${EXTRACT_THRESHOLD}  pool_factor: ${POOL_FACTOR}  merge_gap: ${MERGE_GAP_FRAMES}"
echo "Score lambdas: lam1=${SCORE_LAM_SEMANTIC} lam2=${SCORE_LAM_TEMPORAL} lam3=${SCORE_LAM_COVERAGE} lam4=${SCORE_LAM_COST} lam5=${SCORE_LAM_FEEDBACK} lam6=${SCORE_LAM_CONFIDENCE}"
echo "Output paths:"
echo "  DATA_PATH=${DATA_PATH}"
echo "  PKL_PATH=${PKL_PATH}"
echo "  OTB_ACTION_LIST=${OTB_ACTION_LIST}"
echo "  OTB_VIDEO_ROOT=${OTB_VIDEO_ROOT}"
echo "  EXTRACT_CONTEXT_SEC=${EXTRACT_CONTEXT_SEC}"
echo "  EXTRACT_MIN_CLIP_SEC=${EXTRACT_MIN_CLIP_SEC}"
echo "  FINAL_OUT_DIR=${FINAL_OUT_DIR}"
echo "  S2_OUT_DIR=${S2_OUT_DIR}"
echo "  VIDEO_SEG_ROOT=${VIDEO_SEG_ROOT}"

START_COUNT="$(ls -1 "$FINAL_OUT_DIR"/*.json 2>/dev/null | wc -l || true)"
TARGET_TOTAL="$(
python - "$DATA_PATH" <<'PY'
import json, sys
from pathlib import Path
data_path = Path(sys.argv[1])
if not data_path.is_file():
    print(0); raise SystemExit(0)
raw = data_path.read_text(encoding="utf-8").strip()
data = json.loads(raw) if raw else {}
print(len(data.keys()) if isinstance(data, dict) else 0)
PY
)"
echo "Progress target: ${TARGET_TOTAL} total question(s), ${START_COUNT} already completed"

pids=()
for shard_idx in "${!GPU_ARRAY[@]}"; do
  gpu_id="${GPU_ARRAY[$shard_idx]}"
  if [[ -n "${GEMMA4_GPU_IDS}" && "$NUM_SHARDS" -eq 1 ]]; then
    # Expose every physical device to the process before Python/CUDA init (multi-GPU Gemma).
    cuda_vis="$GEMMA4_GPU_IDS"
    _log_suffix="${GEMMA4_GPU_IDS//,/_}"
    log_file="$LOG_DIR/run_gpt_mstemba_pipeline_otb_gemma${_log_suffix}.log"
  else
    cuda_vis="$gpu_id"
    log_file="$LOG_DIR/run_gpt_mstemba_pipeline_otb_gpu${gpu_id}.log"
  fi
  echo "Starting shard ${shard_idx}/${NUM_SHARDS} CUDA_VISIBLE_DEVICES=${cuda_vis} (log: ${log_file})"
  CUDA_VISIBLE_DEVICES="$cuda_vis" OPENAI_API_KEY="$OPENAI_API_KEY" \
    python TimeProVe_Gemma_GPT.py \
      --data "$DATA_PATH" \
      --classes-file "$OTB_ACTION_LIST" \
      --pkl "$PKL_PATH" \
      --extract-threshold "$EXTRACT_THRESHOLD" \
      --pool-factor "$POOL_FACTOR" \
      --merge-gap-frames "$MERGE_GAP_FRAMES" \
      --score-lam-semantic "$SCORE_LAM_SEMANTIC" \
      --score-lam-temporal "$SCORE_LAM_TEMPORAL" \
      --score-lam-coverage "$SCORE_LAM_COVERAGE" \
      --score-lam-cost "$SCORE_LAM_COST" \
      --score-lam-feedback "$SCORE_LAM_FEEDBACK" \
      --score-lam-confidence "$SCORE_LAM_CONFIDENCE" \
      --otb-video-root "$OTB_VIDEO_ROOT" \
      --extract-context-sec "$EXTRACT_CONTEXT_SEC" \
      --extract-min-clip-sec "$EXTRACT_MIN_CLIP_SEC" \
      --model-path "$MODEL_PATH" \
      "${GEMMA_EXTRA_ARGS[@]}" \
      --gemma4-device-map "$GEMMA4_DEVICE_MAP" \
      --gemma4-dtype "$GEMMA4_DTYPE" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard_idx" \
      --final-out-dir "$FINAL_OUT_DIR" \
      --s2-out "$S2_OUT_DIR" \
      --video-seg-root "$VIDEO_SEG_ROOT" \
      --gpt-vlm-model "$GPT_VLM_MODEL" \
      --gpt-desc-max-frames "$GPT_DESC_MAX_FRAMES" \
      "$@" >"$log_file" 2>&1 &
  pids+=("$!")
done

# ── Progress monitor ────────────────────────────────────────────────────────
monitor_pid=""
if [[ "$TARGET_TOTAL" -gt 0 ]]; then
  python - "$FINAL_OUT_DIR" "$TARGET_TOTAL" "$START_COUNT" "${pids[@]}" <<'PY' &
import glob, os, sys, time
from pathlib import Path

out_dir = Path(sys.argv[1])
total = int(sys.argv[2])
initial = int(sys.argv[3])
pids = [int(x) for x in sys.argv[4:]]

def running(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False

def count_done():
    return len(glob.glob(str(out_dir / "*.json")))

try:
    from tqdm import tqdm
    bar = tqdm(total=total, initial=min(initial, total), desc="Overall progress", dynamic_ncols=True)
    while any(running(pid) for pid in pids):
        bar.n = min(count_done(), total); bar.refresh(); time.sleep(2.0)
    bar.n = min(count_done(), total); bar.refresh(); bar.close()
except Exception:
    while any(running(pid) for pid in pids):
        print(f"[overall] {min(count_done(), total)}/{total}", flush=True); time.sleep(5.0)
    print(f"[overall] {min(count_done(), total)}/{total}", flush=True)
PY
  monitor_pid="$!"
fi

# ── Wait for all shards ─────────────────────────────────────────────────────
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
