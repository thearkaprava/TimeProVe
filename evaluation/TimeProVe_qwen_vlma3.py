#!/usr/bin/env python3
"""
Interpretable VLM pipeline with v10 stage-1 temporal proposals.

Per question:
1) Build ``video_actions_timeline`` from MS-TEMBA/AD pkl scores with a probability
   threshold, then build query-relevance-ordered temporal windows (atomic / query-guided
   LLM merges / context). Neighbor-chain merging is replaced by a single text-only LLM
   call that groups timeline indices that should be considered together to answer the
   query; atomic and local context windows stay as in v10.
2) In order of relevance: a **text-only LLM** proposes a tentative answer from detected
   action labels and timestamps (prioritising actions overlapping the clip window), and that
   text is passed to the VLM as guidance. Then: extract clip -> VLM query-conditioned window
   description -> text-only confidence (sufficient to answer?). If confidence is 1, run
   **final QA** with the VLM again on the same clip, conditioned on the written description,
   and stop. Otherwise continue to the next window.
3) If no window passes confidence, run the same VLM final QA on the last processed
   window's clip and descriptions (fallback).

Optional: --reuse-intermediate cached clips/descriptions; --skip-confidence-refine uses
only the top-relevance window (one clip + VLM QA, no confidence model).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from videollama3 import disable_torch_init

from evaluation.generate_segment_description import (
    discover_videos,
    run_one_video,
    seed_everything,
    tags_from_basename,
)
from evaluation.temporal_segments import (
    ffmpeg_extract_clip_frames,
    get_video_stream_meta,
)
from evaluation.llm_final_answer_query import build_segment_qa_user_content
from evaluation.agent_utils import (
    build_query_guided_action_lookup_order,
    extract_noun_terms,
    extract_verb_terms,
    ActionInterval,
    TemporalWindow,
    _score_temporal_window_for_query,
    build_atomic_windows,
    build_confidence_system_instruction,
    build_confidence_user_message,
    build_context_windows,
    compute_ordered_temporal_windows_for_query,
    deduplicate_windows,
    normalize_actions,
    order_temporal_windows_by_query_relevance,
    parse_confidence_json_from_model_output,
    window_to_dict,
)
from evaluation.llm_text_only_json import build_conversation
from evaluation.pipeline_cleanup import cleanup_s2_intermediates_if_complete
from evaluation.register import INFERENCES

_DEFAULT_DATA = _REPO_ROOT / "data" / "OTB_samples.json"
_DEFAULT_CLASSES = _REPO_ROOT / "data" / "TSU_Action_list.txt"
_DEFAULT_OTB_JSON = _REPO_ROOT / "data" / "smarthome.json"
_DEFAULT_PKL = _REPO_ROOT / "data" / "TSU_best_AD.pkl"
_DEFAULT_OTB_VIDEO_ROOT = Path("/data/vidlab_datasets/smarthome/untrimmed/Videos_mp4")
_DEFAULT_SEG_PARENT = _REPO_ROOT / "workdirs" / "s2_vlm_desc_OTB_pred_tempseg_v10"
_DEFAULT_VIDEO_SEG_ROOT = _DEFAULT_SEG_PARENT / "video_segments"
_DEFAULT_FINAL_DIR = _REPO_ROOT / "workdirs" / "s3_llm_final_ans_OTB_pred_tempseg_v10"
_MAX_REFINE_TRIES_FALLBACK = 10
_MIN_SELECTED_SEGMENT_SEC = 4.0


# --- Disjoint refinement helpers (15Apr26) ---
_MAX_SEGMENT_IOU = 0.20
_STOP_OVERLAP_IOU = 0.50
_MAX_ENDPOINT_DELTA_SEC = 0.05
_CANDIDATE_HINTS = 6


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0.0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    if union <= 0.0:
        return 0.0
    return inter / union


def _build_retry_candidates(
    question: str,
    timeline: list[tuple[str, str, float, float]],
) -> list[dict[str, Any]]:
    terms = extract_verb_terms(question) + extract_noun_terms(question)
    seen_terms: set[str] = set()
    uniq_terms: list[str] = []
    for t in terms:
        if t not in seen_terms:
            seen_terms.add(t)
            uniq_terms.append(t)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, float, str]] = set()
    for code, name, s, e in sorted(timeline, key=lambda x: (x[2], x[3], x[0])):
        key = (float(s), float(e), code)
        if key in seen:
            continue
        seen.add(key)
        lower = name.lower()
        score = sum(1 for t in uniq_terms if t in lower)
        candidates.append(
            {
                "start": float(s),
                "end": float(e),
                "code": code,
                "name": name,
                "score": score,
            }
        )
    candidates.sort(key=lambda x: (-int(x["score"]), float(x["start"]), float(x["end"])))
    return candidates


def _segment_overlaps_history(
    seg: dict[str, Any] | None,
    tried_segments: list[dict[str, float]],
    *,
    iou_thr: float = _MAX_SEGMENT_IOU,
) -> bool:
    if not _segment_dict_has_interval(seg):
        return False
    s = float(seg["start_time"])
    e = float(seg["end_time"])
    for old in tried_segments:
        iou = _interval_iou((s, e), (float(old["start"]), float(old["end"])))
        if iou > iou_thr:
            return True
    return False


def _segment_is_distinct_enough(
    seg: dict[str, Any] | None,
    prev: dict[str, Any] | None,
    tried_segments: list[dict[str, float]],
) -> bool:
    if not _segment_dict_has_interval(seg):
        return False
    if _segment_overlaps_history(seg, tried_segments):
        return False
    if prev is None or not _segment_dict_has_interval(prev):
        return True
    s0, e0 = float(seg["start_time"]), float(seg["end_time"])
    s1, e1 = float(prev["start_time"]), float(prev["end_time"])
    if abs(s0 - s1) <= _MAX_ENDPOINT_DELTA_SEC and abs(e0 - e1) <= _MAX_ENDPOINT_DELTA_SEC:
        return False
    if _interval_iou((s0, e0), (s1, e1)) > _MAX_SEGMENT_IOU:
        return False
    return True


def _dominant_code_for_segment(
    seg: dict[str, Any] | None,
    timeline: list[tuple[str, str, float, float]],
) -> str | None:
    if not _segment_dict_has_interval(seg):
        return None
    s = float(seg["start_time"])
    e = float(seg["end_time"])
    best_code = None
    best_overlap = 0.0
    for code, _name, a_s, a_e in timeline:
        overlap = max(0.0, min(e, a_e) - max(s, a_s))
        if overlap > best_overlap:
            best_overlap = overlap
            best_code = code
    return best_code


def _candidate_fallback_segment(
    *,
    current_segment: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    candidate_cursor: int,
    tried_segments: list[dict[str, float]],
    tried_codes: set[str],
    video_duration: float | None,
) -> tuple[dict[str, Any] | None, int]:
    curr_interval = None
    if _segment_dict_has_interval(current_segment):
        curr_interval = (float(current_segment["start_time"]), float(current_segment["end_time"]))

    # Pass 1: prefer unseen action codes.
    for idx in range(candidate_cursor, len(candidates)):
        cand = candidates[idx]
        seg = {
            "start_time": float(cand["start"]),
            "end_time": float(cand["end"]),
            "justification": (
                f'Disjoint retry candidate selected from timeline: "{cand["name"]}" ({cand["code"]}) '
                f'[{float(cand["start"]):.2f}, {float(cand["end"]):.2f}]'
            ),
        }
        if curr_interval is not None:
            if _interval_iou((seg["start_time"], seg["end_time"]), curr_interval) > _MAX_SEGMENT_IOU:
                continue
        if _segment_overlaps_history(seg, tried_segments):
            continue
        if tried_codes and str(cand["code"]) in tried_codes:
            continue
        return normalize_segment_min_duration(seg, video_duration=video_duration), idx + 1

    # Pass 2: allow seen action codes if nothing else remains.
    for idx in range(candidate_cursor, len(candidates)):
        cand = candidates[idx]
        seg = {
            "start_time": float(cand["start"]),
            "end_time": float(cand["end"]),
            "justification": (
                f'Disjoint retry fallback from timeline: "{cand["name"]}" ({cand["code"]}) '
                f'[{float(cand["start"]):.2f}, {float(cand["end"]):.2f}]'
            ),
        }
        if curr_interval is not None:
            if _interval_iou((seg["start_time"], seg["end_time"]), curr_interval) > _MAX_SEGMENT_IOU:
                continue
        if _segment_overlaps_history(seg, tried_segments):
            continue
        return normalize_segment_min_duration(seg, video_duration=video_duration), idx + 1

    return None, candidate_cursor


def normalize_segment_min_duration(
    seg: dict[str, Any] | None,
    *,
    video_duration: float | None,
    min_sec: float = _MIN_SELECTED_SEGMENT_SEC,
) -> dict[str, Any] | None:
    """
    Enforce a minimum selected segment duration.

    Special-case boundary proposals:
    - If the segment starts at 0 (before-context), keep start at 0 and extend end if needed.
    - If the segment ends at video_duration (after-context), keep end at video end and extend start if needed.
    """
    if not isinstance(seg, dict):
        return seg
    s_raw = seg.get("start_time")
    e_raw = seg.get("end_time")
    try:
        s = float(s_raw) if s_raw is not None else None
        e = float(e_raw) if e_raw is not None else None
    except (TypeError, ValueError):
        return seg
    if s is None or e is None:
        return seg
    if e < s:
        s, e = e, s

    # Clamp to [0, T] when possible.
    s = max(0.0, s)
    if video_duration is not None:
        T = max(0.0, float(video_duration))
        e = max(0.0, min(e, T))
        s = max(0.0, min(s, T))
    else:
        T = None
        e = max(0.0, e)

    if min_sec <= 0:
        out = dict(seg)
        out["start_time"] = s
        out["end_time"] = e
        return out

    dur = e - s
    if dur >= min_sec:
        out = dict(seg)
        out["start_time"] = s
        out["end_time"] = e
        return out

    # If video shorter than min, select whole feasible extent.
    if T is not None and T <= min_sec:
        out = dict(seg)
        out["start_time"] = 0.0
        out["end_time"] = T
        return out

    eps = 1e-3
    is_start_boundary = s <= eps
    is_end_boundary = T is not None and abs(e - T) <= eps

    if is_start_boundary:
        ns = 0.0
        ne = ns + min_sec
        if T is not None:
            ne = min(ne, T)
        out = dict(seg)
        out["start_time"] = ns
        out["end_time"] = ne
        return out

    if is_end_boundary and T is not None:
        ne = T
        ns = max(0.0, ne - min_sec)
        out = dict(seg)
        out["start_time"] = ns
        out["end_time"] = ne
        return out

    # Otherwise expand around the center.
    center = 0.5 * (s + e)
    ns = center - 0.5 * min_sec
    ne = center + 0.5 * min_sec
    if ns < 0.0:
        ne = ne - ns
        ns = 0.0
    if T is not None and ne > T:
        ns = max(0.0, ns - (ne - T))
        ne = T
    out = dict(seg)
    out["start_time"] = ns
    out["end_time"] = ne
    return out


def _segment_dict_has_interval(seg: Any) -> bool:
    if not isinstance(seg, dict):
        return False
    try:
        s_raw, e_raw = seg.get("start_time"), seg.get("end_time")
        if s_raw is None or e_raw is None:
            return False
        s, e = float(s_raw), float(e_raw)
    except (TypeError, ValueError):
        return False
    return e >= s


def fallback_segment_from_ordered_action(
    ordered_actions: list[dict[str, Any]],
    *,
    video_duration: float | None,
) -> dict[str, Any] | None:
    """
    Use the top-ranked OTB action (query-guided order) as the temporal window:
    the segment where that named action is performed in the video.
    """
    if not ordered_actions:
        return None
    first = ordered_actions[0]
    try:
        s = float(first["start_sec"])
        e = float(first["end_sec"])
    except (KeyError, TypeError, ValueError):
        return None
    if e < s:
        s, e = e, s
    code = str(first.get("code", ""))
    name = str(first.get("name", ""))
    seg = {
        "start_time": s,
        "end_time": e,
        "justification": (
            f'Fallback segment: query-ranked action "{name}" ({code}) '
            f"[{s:.2f}, {e:.2f}] (no other segment selected)."
        ),
    }
    return normalize_segment_min_duration(seg, video_duration=video_duration)


def fallback_segment_from_timeline_earliest(
    video_actions_timeline: list[tuple[str, str, float, float]],
    *,
    video_duration: float | None,
) -> dict[str, Any] | None:
    """Last resort: earliest action instance in the GT timeline."""
    if not video_actions_timeline:
        return None
    code, name, s, e = sorted(video_actions_timeline, key=lambda x: (x[2], x[3], x[0]))[0]
    seg = {
        "start_time": float(s),
        "end_time": float(e),
        "justification": (
            f'Fallback segment: earliest timeline action "{name}" ({code}) '
            f"[{float(s):.2f}, {float(e):.2f}]."
        ),
    }
    return normalize_segment_min_duration(seg, video_duration=video_duration)


def _window_dict_to_segment(
    window: dict[str, Any],
    *,
    video_duration: float | None,
) -> dict[str, Any]:
    """Convert a v10 temporal window dict to start_time/end_time segment for clipping."""
    meta = window.get("metadata") or {}
    wt = str(window.get("window_type", ""))
    r = meta.get("relevance_rank", "")
    return normalize_segment_min_duration(
        {
            "start_time": float(window["start"]),
            "end_time": float(window["end"]),
            "justification": (
                f'v10 stage-1 window rank {r} ({wt}): '
                f'{meta.get("start_action_label", "")} → {meta.get("end_action_label", "")}'
            ),
        },
        video_duration=video_duration,
    )


def ensure_temporal_segment(
    proposal: Any,
    *,
    ordered_actions: list[dict[str, Any]],
    video_actions_timeline: list[tuple[str, str, float, float]],
    video_duration: float | None,
) -> dict[str, Any]:
    """
    Always return a dict with a valid [start_time, end_time] when the timeline is non-empty.
    Prefer the given proposal when it parses; otherwise use top ordered action, then earliest timeline.
    """
    cand = proposal if isinstance(proposal, dict) else None
    if _segment_dict_has_interval(cand):
        out = normalize_segment_min_duration(cand, video_duration=video_duration)
        if isinstance(out, dict) and _segment_dict_has_interval(out):
            return out
    fb = fallback_segment_from_ordered_action(
        ordered_actions, video_duration=video_duration
    )
    if fb is not None and _segment_dict_has_interval(fb):
        return fb
    fb2 = fallback_segment_from_timeline_earliest(
        video_actions_timeline, video_duration=video_duration
    )
    if fb2 is not None and _segment_dict_has_interval(fb2):
        return fb2
    return normalize_segment_min_duration(
        {
            "start_time": 0.0,
            "end_time": max(0.0, float(video_duration or 0.0)),
            "justification": "Fallback: full video span (no action intervals available).",
        },
        video_duration=video_duration,
    )


def run_text_only(
    model: Any,
    processor: Any,
    mm_infer_fn: Any,
    instruction: str,
    user_text: str,
    gen_kwargs: dict[str, Any],
) -> str:
    conversation = build_conversation(instruction, user_text)
    inputs = processor(
        images=None,
        text=conversation,
        merge_size=1,
        return_tensors="pt",
    )
    return mm_infer_fn(
        inputs,
        model=model,
        tokenizer=processor.tokenizer,
        modal="text",
        **gen_kwargs,
    )


def _build_vlm_final_answer_prompt(
    question: str,
    query_conditioned_descriptions: list[str],
    system_instruction: str,
) -> str:
    """
    Text paired with the video for the second VLM pass: answer the query using both
    the clip pixels and the earlier query-conditioned descriptions of this window.
    """
    parts: list[str] = []
    if system_instruction.strip():
        parts.append(system_instruction.strip())
        parts.append("")
    parts.extend(
        [
            "You will watch a short video clip again.",
            "Below is a query-conditioned description of this same clip that was written earlier "
            "(it focuses on details relevant to the question).",
            "Use BOTH what you see in the video and the description to answer the question.",
            "If the description and the video disagree on a visible fact, trust the video.",
            "",
        ]
    )
    for i, t in enumerate(query_conditioned_descriptions, start=1):
        parts.append(f"Query-conditioned description {i}:\n{t.strip()}")
        parts.append("")
    parts.append(f'Question:\n"{question.strip()}"')
    parts.append("")
    parts.append(
        "Answer the question directly and concisely. Do not repeat the full description unless "
        "needed for clarity."
    )
    return "\n".join(parts).strip()


def run_vlm_final_answer_from_clip(
    clip_path: Path,
    *,
    question: str,
    description_texts: list[str],
    system_instruction: str,
    model: Any,
    processor: Any,
    mm_infer_fn: Any,
    fps: int,
    max_frames: int,
    gen_kwargs: dict[str, Any],
) -> str:
    """Second multimodal pass: same clip + query-conditioned text -> direct answer."""
    path_str = str(clip_path.resolve())
    frames, timestamps = processor.load_video(
        path_str,
        start_time=None,
        end_time=None,
        precise_time=True,
        fps=fps,
        max_frames=max_frames,
    )
    image_inputs = processor.process_images([frames], merge_size=2, return_tensors="pt")
    text_prompt = _build_vlm_final_answer_prompt(
        question, description_texts, system_instruction
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "num_frames": len(timestamps),
                    "timestamps": timestamps,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    text_input = processor.process_text(
        prompt,
        image_inputs,
        padding=False,
        padding_side=None,
        return_tensors="pt",
    )
    data_dict = {**image_inputs, **text_input}
    infer_kw = {**gen_kwargs}
    infer_kw.setdefault("do_sample", False)
    return str(
        mm_infer_fn(
            data_dict,
            model=model,
            tokenizer=processor.tokenizer,
            modal="video",
            **infer_kw,
        )
    ).strip()


_LLM_TIMELINE_INITIAL_ANSWER_SYSTEM = (
    "You answer questions about a video using ONLY automatically detected action labels "
    "and their approximate start and end times in seconds (from an action recognition model). "
    "You do NOT see video pixels or hear audio. Give one concise tentative answer to the question. "
    "If the timeline is ambiguous or clearly insufficient, say so in one short sentence. "
    "Plain text only — no JSON and no markdown code fences."
)


def _timeline_actions_overlapping_window(
    video_actions_timeline: list[tuple[str, str, float, float]],
    w_start: float,
    w_end: float,
) -> list[tuple[str, str, float, float]]:
    out: list[tuple[str, str, float, float]] = []
    for row in video_actions_timeline:
        try:
            code, name, s, e = row[0], row[1], float(row[2]), float(row[3])
        except (IndexError, TypeError, ValueError):
            continue
        if max(0.0, min(w_end, e) - max(w_start, s)) > 0.0:
            out.append((code, name, s, e))
    return sorted(out, key=lambda x: (x[2], x[3], str(x[0])))


def _build_llm_timeline_initial_answer_user_message(
    *,
    question: str,
    video_id: str,
    video_duration: float | None,
    w_start: float,
    w_end: float,
    video_actions_timeline: list[tuple[str, str, float, float]],
    global_head_max: int,
) -> str:
    lines = [
        f"Video ID: {video_id}",
        f"Evidence window under consideration (seconds): [{w_start:.2f}, {w_end:.2f}]",
    ]
    if video_duration is not None:
        lines.append(f"Approximate full video duration (seconds): {video_duration:.2f}")
    lines.append("")
    overlap = _timeline_actions_overlapping_window(video_actions_timeline, w_start, w_end)
    if overlap:
        lines.append("Detected actions overlapping this window (code — name [start_sec, end_sec]):")
        for code, name, s, e in overlap:
            lines.append(f"  {code} — {name} [{s:.2f}, {e:.2f}]")
    else:
        n = len(video_actions_timeline)
        m = min(n, global_head_max) if global_head_max > 0 else n
        head = video_actions_timeline[:m]
        if n:
            lines.append(
                "No detector segments strictly overlapped the window; showing the earliest part of the "
                "full-video chronological timeline for context."
            )
            lines.append(f"First {len(head)} chronological actions (code — name [start_sec, end_sec]):")
            for code, name, s, e in head:
                lines.append(f"  {code} — {name} [{s:.2f}, {e:.2f}]")
            if n > m:
                lines.append(f"(Timeline truncated for prompt: {n} actions total.)")
        else:
            lines.append("Empty action timeline (no detections above threshold).")
    lines.extend(["", f"Question:\n{question.strip()}", "", "Tentative answer (from labels and times only):"])
    return "\n".join(lines).strip()


def run_llm_timeline_initial_answer(
    *,
    question: str,
    video_id: str,
    video_duration: float | None,
    w_start: float,
    w_end: float,
    video_actions_timeline: list[tuple[str, str, float, float]],
    model: Any,
    processor: Any,
    mm_infer_fn: Any,
    gen_common: dict[str, Any],
    max_new_tokens: int,
    global_head_max: int,
) -> tuple[str, str]:
    """Text-only tentative answer from predicted actions and segment times. Returns (answer, raw)."""
    user = _build_llm_timeline_initial_answer_user_message(
        question=question,
        video_id=video_id,
        video_duration=video_duration,
        w_start=w_start,
        w_end=w_end,
        video_actions_timeline=video_actions_timeline,
        global_head_max=global_head_max,
    )
    gen = {**gen_common, "max_new_tokens": max_new_tokens}
    raw = run_text_only(
        model,
        processor,
        mm_infer_fn,
        _LLM_TIMELINE_INITIAL_ANSWER_SYSTEM,
        user,
        gen,
    )
    return raw.strip(), raw


_LLM_TIMELINE_MERGE_SYSTEM = (
    "You decide which detected actions from the numbered list should be merged into one temporal "
    "evidence span to answer the question. "
    "Output a single JSON object only, no markdown fences or commentary. "
    "Schema: {\"merge_groups\": [ {\"action_indices\": [int, ...], \"why\": string } ]}. "
    "Each entry lists 0-based indices that belong together for answering the query; you may merge "
    "non-neighboring indices if the question calls for joint evidence. "
    "Include only groups with at least two indices; omit unrelated actions and omit singletons."
)


def _timeline_head_for_llm_prompt(
    video_actions_timeline: list[tuple[str, str, float, float]],
    max_actions: int,
) -> tuple[int, bool]:
    """Returns (M, truncated) where M = min(len, max_actions)."""
    n = len(video_actions_timeline)
    m = min(n, max_actions)
    return m, m < n


def _build_llm_timeline_merge_user_message(
    *,
    question: str,
    video_id: str,
    video_duration: float | None,
    indexed_actions: list[tuple[int, str, str, float, float]],
    truncated_from_full: bool,
) -> str:
    lines = [
        f"Video ID: {video_id}",
        f"Question: {question}",
    ]
    if video_duration is not None:
        lines.append(f"Video duration (seconds): {video_duration:.2f}")
    if truncated_from_full:
        lines.append(
            "(Note: only the earliest part of the action timeline is listed below due to length; "
            "indices are 0-based within this list.)"
        )
    lines.extend(["", "Numbered detected actions (index: code — name [start_sec, end_sec]):"])
    for i, code, name, s, e in indexed_actions:
        lines.append(f"  {i}: {code} — {name} [{s:.2f}, {e:.2f}]")
    lines.extend(
        [
            "",
            f"N = {len(indexed_actions)} (valid indices 0 … {len(indexed_actions) - 1}).",
            "Return JSON as specified in the system message.",
        ]
    )
    return "\n".join(lines)


def _parse_merge_groups_from_llm(text: str) -> list[list[int]]:
    """Parse merge_groups from model output; invalid/missing entries become []."""
    if not text or not text.strip():
        return []
    raw = text.strip()
    candidates = [raw]
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        candidates.insert(0, m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        mg = obj.get("merge_groups")
        if mg is None:
            mg = obj.get("groups")
        if not isinstance(mg, list):
            continue
        groups: list[list[int]] = []
        for item in mg:
            idxs: list[int] = []
            if isinstance(item, dict):
                arr = item.get("action_indices")
                if arr is None:
                    arr = item.get("indices")
            elif isinstance(item, list):
                arr = item
            else:
                continue
            if not isinstance(arr, list):
                continue
            for x in arr:
                try:
                    idxs.append(int(x))
                except (TypeError, ValueError):
                    pass
            if idxs:
                groups.append(idxs)
        return groups
    return []


def _union_find_merge_groups(raw_groups: list[list[int]], n: int) -> list[list[int]]:
    """Turn possibly overlapping LLM groups into disjoint merged components (valid indices only)."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in raw_groups:
        valid = sorted({i for i in group if 0 <= i < n})
        if len(valid) < 2:
            continue
        a0 = valid[0]
        for b in valid[1:]:
            union(a0, b)

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        buckets.setdefault(r, []).append(i)
    return [sorted(v) for v in buckets.values()]


def _windows_from_merge_components(
    normalized: list[ActionInterval],
    components: list[list[int]],
) -> list[TemporalWindow]:
    """Build TemporalWindow list (window_type merged) from disjoint index components."""
    out: list[TemporalWindow] = []
    for comp in components:
        if not comp:
            continue
        ii = sorted(comp)
        labels = [normalized[j].label for j in ii]
        start = min(normalized[j].start for j in ii)
        end = max(normalized[j].end for j in ii)
        dur = end - start
        meta = {
            "num_actions": len(ii),
            "source_query": "",
            "start_action_label": labels[0],
            "end_action_label": labels[-1],
            "merge_policy": "llm_query_evidence",
        }
        out.append(
            TemporalWindow(
                start=start,
                end=end,
                action_indices=ii,
                action_labels=labels,
                window_type="merged",
                duration=dur,
                metadata=meta,
            )
        )
    return out


def compute_ordered_temporal_windows_llm_query_merge(
    query: str,
    video_actions_timeline: list[tuple[str, str, float, float]],
    video_duration: float | None,
    *,
    model: Any,
    processor: Any,
    mm_infer_fn: Any,
    gen_common: dict[str, Any],
    merge_max_new_tokens: int,
    max_actions_in_prompt: int,
    video_id: str,
    log: logging.Logger,
) -> tuple[list[dict[str, Any]], str]:
    """
    Like v10's compute_ordered_temporal_windows_for_query but replaces neighbor-chain
    merged windows with a single LLM pass over the indexed timeline (first M actions).
    """
    if not video_actions_timeline:
        return [], ""
    raw_actions_full = [
        {"label": name, "start_sec": float(s), "end_sec": float(e)}
        for _code, name, s, e in video_actions_timeline
    ]
    normalized_full = normalize_actions(raw_actions_full)
    if not normalized_full:
        return [], ""
    n_full = len(normalized_full)
    m_show, truncated = _timeline_head_for_llm_prompt(video_actions_timeline, max_actions_in_prompt)
    vl = (
        float(video_duration)
        if video_duration is not None
        else max(float(e) for _c, _n, _s, e in video_actions_timeline)
    )
    head_tl = video_actions_timeline[:m_show]
    indexed = [
        (i, code, name, float(s), float(e))
        for i, (code, name, s, e) in enumerate(head_tl)
    ]
    user_msg = _build_llm_timeline_merge_user_message(
        question=query,
        video_id=video_id,
        video_duration=video_duration,
        indexed_actions=indexed,
        truncated_from_full=truncated,
    )
    gen_merge = {**gen_common, "max_new_tokens": merge_max_new_tokens}
    raw_out = run_text_only(
        model, processor, mm_infer_fn, _LLM_TIMELINE_MERGE_SYSTEM, user_msg, gen_merge
    )
    parsed = _parse_merge_groups_from_llm(raw_out)
    components = _union_find_merge_groups(parsed, m_show)
    atomic = build_atomic_windows(normalized_full)
    context = build_context_windows(normalized_full, vl)
    multi_only = [c for c in components if len(c) >= 2]
    query_merged = _windows_from_merge_components(normalized_full, multi_only)
    combined = atomic + context + query_merged
    for w in combined:
        w.metadata["source_query"] = query
    wins = deduplicate_windows(combined)
    if not wins:
        return [], raw_out
    ordered = order_temporal_windows_by_query_relevance(query, wins)
    for i, w in enumerate(ordered, start=1):
        w.metadata["relevance_rank"] = i
        w.metadata["query_lexical_score"] = _score_temporal_window_for_query(w, query)
    if truncated:
        log.warning(
            "LLM timeline merge: prompt shows first %d / %d actions for video_id=%s",
            m_show,
            n_full,
            video_id,
        )
    return [window_to_dict(w) for w in ordered], raw_out


def load_otb_classes(path: Path) -> list[tuple[str, str, str]]:
    """
    Parse OTB action-list file (format: ``index<TAB>name``).

    Returns a list of ``(class_code, numeric_id_str, name)`` in index order,
    matching the convention used by ``load_otb_classes``.
    """
    out: list[tuple[str, str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) < 2:
            parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        idx_str, name = parts[0].strip(), parts[1].strip()
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        code = f"c{idx:03d}"
        out.append((code, str(idx), name))
    return out


def get_predicted_actions_from_pkl(
    video_id: str,
    pkl_data: dict,
    class_names: list[str],
    threshold: float,
    video_path: Path,
    merge_gap_frames: int = 1,
    pool_factor: int = 16,
) -> tuple[list[tuple[str, str, float, float]], float | None]:
    """
    Build ``video_actions_timeline`` from MS-TEMBA/AD frame-level predictions.

    The pkl stores **raw action-detection probabilities** (already in [0, 1],
    no sigmoid needed) of shape ``(num_classes, num_pooled_frames)``.  The
    temporal dimension is pooled with a sliding window of ``pool_factor``
    frames, so pkl frame index ``i`` corresponds to original video frames
    ``[i * pool_factor, (i+1) * pool_factor)``.

    Frame-to-time conversion
    ------------------------
    Given video FPS ``f`` obtained from ``get_video_stream_meta``::

        start_sec = pkl_start_frame * pool_factor / f
        end_sec   = pkl_end_frame   * pool_factor / f

    Frames where ``prob >= threshold`` are considered active for that class.
    Contiguous runs are merged into intervals; same-class intervals separated
    by at most ``merge_gap_frames`` pooled frames are fused together.

    Returns
    -------
    timeline
        List of ``(class_code, class_name, start_sec, end_sec)`` tuples sorted
        by start time, ready for use as ``video_actions_timeline``.
    video_duration_sec
        Total video duration in seconds derived from video metadata, or
        ``None`` if it could not be determined.
    """
    if video_id not in pkl_data:
        return [], None

    num_classes = len(class_names)
    arr = np.asarray(pkl_data[video_id], dtype=np.float64)
    if arr.ndim != 2:
        return [], None

    # Normalise to (T, C): first dim is num_classes when shape[0] == num_classes.
    if arr.shape[0] == num_classes and arr.shape[1] != num_classes:
        arr = arr.T
    elif arr.shape[1] != num_classes and arr.shape[0] != num_classes:
        arr = arr.T if arr.shape[0] > arr.shape[1] else arr

    T = arr.shape[0]
    num_cls = min(arr.shape[1], num_classes)

    # Values are already probabilities in [0, 1] — no sigmoid transformation.
    probs = arr[:, :num_cls]

    # Derive FPS and video duration from actual video metadata.
    video_duration_sec: float | None = None
    fps_actual: float | None = None
    try:
        vmeta = get_video_stream_meta(video_path)
        fps_num = vmeta.get("fps_num") or 0
        fps_den = vmeta.get("fps_den") or 1
        nb_frames = vmeta.get("nb_frames")
        if fps_den > 0 and fps_num > 0:
            fps_actual = float(fps_num) / float(fps_den)
            if nb_frames and nb_frames > 0:
                video_duration_sec = float(nb_frames) / fps_actual
    except Exception:
        pass

    # Decode contiguous active-frame segments per class.
    raw_actions: list[dict] = []
    for c in range(num_cls):
        mask = probs[:, c] >= threshold
        i = 0
        while i < T:
            if not mask[i]:
                i += 1
                continue
            j = i + 1
            while j < T and mask[j]:
                j += 1
            raw_actions.append(
                {
                    "class_id": c,
                    "class_name": class_names[c],
                    "start_frame": i,   # pooled frame index
                    "end_frame": j,     # pooled frame index (exclusive)
                }
            )
            i = j

    # Merge same-class segments that are close together.
    if merge_gap_frames >= 0 and raw_actions:
        by_class: dict[int, list[dict]] = {}
        for a in raw_actions:
            by_class.setdefault(a["class_id"], []).append(a)
        merged: list[dict] = []
        for cid in sorted(by_class.keys()):
            segs = sorted(by_class[cid], key=lambda x: x["start_frame"])
            cur = dict(segs[0])
            for nxt in segs[1:]:
                if nxt["start_frame"] - cur["end_frame"] <= merge_gap_frames:
                    cur["end_frame"] = nxt["end_frame"]
                else:
                    merged.append(cur)
                    cur = dict(nxt)
            merged.append(cur)
        raw_actions = sorted(merged, key=lambda x: (x["start_frame"], x["class_id"]))

    # Convert pooled-frame indices to seconds.
    # pkl_frame_index × pool_factor = actual video frame number.
    # actual video frame / fps = time in seconds.
    timeline: list[tuple[str, str, float, float]] = []
    for act in raw_actions:
        code = f"c{act['class_id']:03d}"
        name = act["class_name"]
        sf, ef = act["start_frame"], act["end_frame"]
        if fps_actual is not None and fps_actual > 0:
            start_sec = sf * pool_factor / fps_actual
            end_sec = ef * pool_factor / fps_actual
        elif video_duration_sec is not None and T > 0:
            # Fallback: proportional mapping (equivalent when T = nb_frames / pool_factor).
            start_sec = sf * video_duration_sec / T
            end_sec = ef * video_duration_sec / T
        else:
            start_sec = float(sf * pool_factor)
            end_sec = float(ef * pool_factor)
        timeline.append((code, name, start_sec, end_sec))

    return timeline, video_duration_sec


def get_predicted_actions_with_confidence_from_pkl(
    video_id: str,
    pkl_data: dict,
    class_names: list[str],
    threshold: float,
    video_path: Path,
    merge_gap_frames: int = 1,
    pool_factor: int = 16,
) -> tuple[list[tuple[str, str, float, float, float]], float | None]:
    """
    Like get_predicted_actions_from_pkl but also records the mean detection
    confidence (mean probability over active frames) for each merged segment.

    Returns
    -------
    timeline_with_conf
        List of ``(class_code, class_name, start_sec, end_sec, mean_confidence)``
        tuples sorted by start time.  ``mean_confidence`` is the mean of the
        raw probabilities over all pooled frames that were active (>= threshold)
        for that merged segment, so it is always in [threshold, 1].
    video_duration_sec
        Total video duration in seconds derived from video metadata, or None.
    """
    if video_id not in pkl_data:
        return [], None

    num_classes = len(class_names)
    arr = np.asarray(pkl_data[video_id], dtype=np.float64)
    if arr.ndim != 2:
        return [], None

    if arr.shape[0] == num_classes and arr.shape[1] != num_classes:
        arr = arr.T
    elif arr.shape[1] != num_classes and arr.shape[0] != num_classes:
        arr = arr.T if arr.shape[0] > arr.shape[1] else arr

    T = arr.shape[0]
    num_cls = min(arr.shape[1], num_classes)
    probs = arr[:, :num_cls]

    video_duration_sec: float | None = None
    fps_actual: float | None = None
    try:
        vmeta = get_video_stream_meta(video_path)
        fps_num = vmeta.get("fps_num") or 0
        fps_den = vmeta.get("fps_den") or 1
        nb_frames = vmeta.get("nb_frames")
        if fps_den > 0 and fps_num > 0:
            fps_actual = float(fps_num) / float(fps_den)
            if nb_frames and nb_frames > 0:
                video_duration_sec = float(nb_frames) / fps_actual
    except Exception:
        pass

    raw_actions: list[dict] = []
    for c in range(num_cls):
        mask = probs[:, c] >= threshold
        i = 0
        while i < T:
            if not mask[i]:
                i += 1
                continue
            j = i + 1
            while j < T and mask[j]:
                j += 1
            seg_probs = probs[i:j, c]
            raw_actions.append(
                {
                    "class_id": c,
                    "class_name": class_names[c],
                    "start_frame": i,
                    "end_frame": j,
                    "mean_confidence": float(np.mean(seg_probs)),
                }
            )
            i = j

    if merge_gap_frames >= 0 and raw_actions:
        by_class: dict[int, list[dict]] = {}
        for a in raw_actions:
            by_class.setdefault(a["class_id"], []).append(a)
        merged: list[dict] = []
        for cid in sorted(by_class.keys()):
            segs = sorted(by_class[cid], key=lambda x: x["start_frame"])
            cur = dict(segs[0])
            for nxt in segs[1:]:
                if nxt["start_frame"] - cur["end_frame"] <= merge_gap_frames:
                    span = probs[cur["start_frame"]:nxt["end_frame"], cid]
                    active = span[span >= threshold]
                    cur["end_frame"] = nxt["end_frame"]
                    cur["mean_confidence"] = float(np.mean(active)) if len(active) > 0 else cur["mean_confidence"]
                else:
                    merged.append(cur)
                    cur = dict(nxt)
            merged.append(cur)
        raw_actions = sorted(merged, key=lambda x: (x["start_frame"], x["class_id"]))

    timeline_with_conf: list[tuple[str, str, float, float, float]] = []
    for act in raw_actions:
        code = f"c{act['class_id']:03d}"
        name = act["class_name"]
        sf, ef = act["start_frame"], act["end_frame"]
        if fps_actual is not None and fps_actual > 0:
            start_sec = sf * pool_factor / fps_actual
            end_sec = ef * pool_factor / fps_actual
        elif video_duration_sec is not None and T > 0:
            start_sec = sf * video_duration_sec / T
            end_sec = ef * video_duration_sec / T
        else:
            start_sec = float(sf * pool_factor)
            end_sec = float(ef * pool_factor)
        timeline_with_conf.append((code, name, start_sec, end_sec, act["mean_confidence"]))

    return timeline_with_conf, video_duration_sec


def run_extract_gt_timeline(
    video_path: Path,
    video_key: str,
    timeline: list[tuple[str, str, float, float]],
    out_dir: Path,
    context_sec: float = 0.0,
    min_clip_sec: float = 0.0,
) -> None:
    """
    Extract segments directly from GT (start_sec, end_sec) windows.
    One clip per timeline entry.
    """
    vmeta = get_video_stream_meta(video_path)
    nb_frames = vmeta["nb_frames"]
    if nb_frames is None:
        raise RuntimeError(
            f"extract_gt_timeline failed: nb_frames missing for {video_path}. "
            "Use source videos that expose frame count via ffprobe."
        )
    fps_num, fps_den = vmeta["fps_num"], vmeta["fps_den"]
    if fps_den == 0:
        raise RuntimeError(f"extract_gt_timeline failed: invalid fps denominator for {video_path}")
    fps = float(fps_num) / float(fps_den)
    if fps <= 0:
        raise RuntimeError(f"extract_gt_timeline failed: non-positive fps for {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    context_frames = max(0, int(math.ceil(context_sec * fps)))
    min_frames = max(1, int(math.ceil(min_clip_sec * fps)))

    for seg_i, (_code, _name, start_sec, end_sec) in enumerate(timeline):
        if end_sec < start_sec:
            continue
        sf = int(math.floor(start_sec * fps))
        ef = int(math.ceil(end_sec * fps) - 1)
        sf = max(0, min(nb_frames - 1, sf))
        ef = max(sf, min(nb_frames - 1, ef))

        # ffmpeg_extract_clip_frames uses half-open [start_frame, end_frame_excl); sf..ef are inclusive.
        start_frame = sf
        end_excl = ef + 1

        # Add configurable temporal context around GT action boundaries.
        if context_frames > 0:
            start_frame = max(0, start_frame - context_frames)
            end_excl = min(nb_frames, end_excl + context_frames)

        # Keep clips decodable even when GT windows are near-instant.
        if end_excl <= start_frame:
            end_excl = min(nb_frames, start_frame + 1)

        # Enforce a minimum clip duration so the action is visible to the VLM.
        curr_len = end_excl - start_frame
        if curr_len < min_frames:
            missing = min_frames - curr_len
            left = missing // 2
            right = missing - left
            start_frame = max(0, start_frame - left)
            end_excl = min(nb_frames, end_excl + right)
            if end_excl - start_frame < min_frames:
                if start_frame == 0:
                    end_excl = min(nb_frames, start_frame + min_frames)
                elif end_excl == nb_frames:
                    start_frame = max(0, end_excl - min_frames)

        clip_path = out_dir / f"{video_key}_seg{seg_i:02d}_{start_sec:.2f}-{end_sec:.2f}.mp4"
        ffmpeg_extract_clip_frames(video_path, start_frame, end_excl, fps_num, fps_den, clip_path)


def build_segment_descriptions_payload(
    question_id: str,
    video_dir: Path,
    model: Any,
    processor: Any,
    mm_infer_fn: Any,
    fps: int,
    max_frames: int,
    max_new_tokens: int,
    model_path: str,
    glob_pat: str,
    tag_delimiter: str,
    question: str,
    prompt_style: str,
    *,
    action_context: str | None = None,
) -> dict[str, Any]:
    video_paths = discover_videos(str(video_dir), glob_pat)
    descriptions: dict[str, Any] = {}
    for video_path in video_paths:
        basename = os.path.basename(video_path)
        meta = tags_from_basename(basename, tag_delimiter)
        try:
            description = run_one_video(
                video_path,
                model,
                processor,
                mm_infer_fn,
                fps,
                max_frames,
                max_new_tokens,
                question=question,
                prompt_style=prompt_style,
                action_context=action_context,
            )
        except Exception as e:
            description = f"ERROR: {e}"
        descriptions[basename] = {
            "video_path": os.path.abspath(video_path),
            "description": description,
            **meta,
        }
    out: dict[str, Any] = {
        "question_id": question_id,
        "question": question,
        "model_path": model_path,
        "video_dir": str(video_dir.resolve()),
        "glob": glob_pat,
        "fps": fps,
        "max_frames": max_frames,
        "prompt_style": prompt_style,
        "action_context": action_context,
        "descriptions": descriptions,
    }
    return out


def _normalize_answer_for_vote(answer: str) -> str:
    """
    Canonicalize free-form answers for majority voting.
    We keep this intentionally lightweight so nearby paraphrases still match.
    """
    text = answer.strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip().lower()
    first_line = re.sub(r"[`\"']", "", first_line)
    first_line = re.sub(r"[^a-z0-9]+", " ", first_line)
    first_line = re.sub(r"\b(a|an|the)\b", " ", first_line)
    return re.sub(r"\s+", " ", first_line).strip()


def _majority_vote_topk(action_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    ballots: list[dict[str, Any]] = []
    for i, pred in enumerate(action_predictions):
        if pred.get("error"):
            continue
        raw = pred.get("output")
        if not isinstance(raw, str) or not raw.strip():
            continue
        vote_key = _normalize_answer_for_vote(raw)
        if not vote_key:
            continue
        ballots.append(
            {
                "ballot_index": i,
                "lookup_rank": int(pred.get("lookup_rank", i + 1)),
                "vote_key": vote_key,
                "raw_answer": raw.strip(),
                "action_code": pred.get("code"),
            }
        )

    if not ballots:
        return {
            "method": "majority_vote_topk_actions",
            "final_answer": None,
            "vote_counts": {},
            "winning_vote_key": None,
            "num_valid_votes": 0,
            "tie_breaker": "highest_importance_action_rank",
        }

    counts = Counter(b["vote_key"] for b in ballots)
    max_votes = max(counts.values())
    winning_keys = {k for k, v in counts.items() if v == max_votes}
    winner = sorted(
        (b for b in ballots if b["vote_key"] in winning_keys),
        key=lambda x: (int(x["lookup_rank"]), int(x["ballot_index"])),
    )[0]

    return {
        "method": "majority_vote_topk_actions",
        "final_answer": winner["raw_answer"],
        "vote_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "winning_vote_key": winner["vote_key"],
        "num_valid_votes": len(ballots),
        "winner_source": {
            "lookup_rank": winner["lookup_rank"],
            "action_code": winner["action_code"],
        },
        "tie_breaker": "highest_importance_action_rank",
    }


def _collect_valid_descriptions(desc_payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    desc_map = desc_payload.get("descriptions")
    if not isinstance(desc_map, dict):
        return [], []
    texts: list[str] = []
    seg_keys: list[str] = []
    for k in sorted(desc_map.keys()):
        item = desc_map[k]
        if not isinstance(item, dict):
            continue
        d = item.get("description")
        if isinstance(d, str) and d.strip() and not str(d).startswith("ERROR:"):
            seg_keys.append(k)
            texts.append(d.strip())
    return texts, seg_keys


def _append_unique_descriptions(
    texts: list[str],
    seg_keys: list[str],
    new_texts: list[str],
    new_seg_keys: list[str],
) -> tuple[list[str], list[str]]:
    seen = set(zip(texts, seg_keys))
    for t, k in zip(new_texts, new_seg_keys):
        pair = (t, k)
        if pair in seen:
            continue
        seen.add(pair)
        texts.append(t)
        seg_keys.append(k)
    return texts, seg_keys


def _description_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    return {w for w in words if len(w) >= 3}


def _normalize_answer_text(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,!?:;\"'")


def _answer_found_in_descriptions(gt_answer: Any, description_texts: list[str]) -> bool:
    gt = _normalize_answer_text(gt_answer)
    if not gt:
        return False
    for t in description_texts:
        if gt in _normalize_answer_text(t):
            return True
    return False


def _rerank_remaining_windows_by_descriptions(
    remaining_windows: list[dict[str, Any]],
    description_texts: list[str],
    question: str,
    *,
    reason_terms: set[str] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Re-rank remaining windows after a failed confidence round.

    Sort key (descending), in priority order:
      1. reason_overlap  — content words from the LLM's confidence reason that appear in
                           the window's action labels. These directly name what was missing,
                           so windows that contain those actions are tried first.
      2. desc_overlap    — content words from accumulated descriptions that appear in the
                           window's action labels. Surfaces windows contextually related to
                           what has already been seen.
      3. query_overlap   — noun/verb terms from the question that appear in the window's
                           action labels (same as the original static ranking signal).
      4. -prior_rank     — tie-breaker: preserve the original v10 relevance ordering.
    """
    if not remaining_windows or not description_texts:
        return remaining_windows, False

    desc_terms: set[str] = set()
    for t in description_texts:
        desc_terms.update(_description_terms(t))
    if not desc_terms:
        return remaining_windows, False

    question_terms = set(extract_noun_terms(question) + extract_verb_terms(question))
    question_terms = {t.lower() for t in question_terms if isinstance(t, str)}

    scored: list[tuple[int, int, int, int, dict[str, Any]]] = []
    for idx, w in enumerate(remaining_windows):
        labels = [str(x) for x in (w.get("action_labels") or [])]
        meta = w.get("metadata") or {}
        label_blob = " ".join(
            labels
            + [str(meta.get("start_action_label", "")), str(meta.get("end_action_label", ""))]
        ).lower()
        label_terms = _description_terms(label_blob)
        reason_overlap = len(reason_terms & label_terms) if reason_terms else 0
        desc_overlap = len(desc_terms & label_terms)
        query_overlap = len(question_terms & label_terms)
        prior_rank = int(meta.get("relevance_rank", 10**6))
        scored.append((reason_overlap, desc_overlap, query_overlap, -prior_rank, w))

    re_ranked = [w for _a, _b, _c, _d, w in sorted(scored, key=lambda x: x[:4], reverse=True)]
    changed = any(id(a) != id(b) for a, b in zip(re_ranked, remaining_windows))
    return re_ranked, changed


# ---------------------------------------------------------------------------
# Relevance scoring: R(w | q) = λ1·R_sem + λ2·R_tmp + λ3·R_cov − λ4·R_cost
#                              + λ5·R_feedback  (active only after a failed round)
# ---------------------------------------------------------------------------

_TEMPORAL_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("before",  [r"\bbefore\b"]),
    ("after",   [r"\bafter\b"]),
    ("between", [r"\bbetween\b", r"\bfrom\b.{1,60}\bto\b"]),
    ("first",   [r"\bfirst\b", r"\binitially\b", r"\bat the (start|beginning)\b"]),
    ("last",    [r"\blast\b", r"\bfinally\b", r"\bat the end\b"]),
    ("state",   [r"\bwhat (is|was|are|were)\b", r"\bduring\b", r"\bwhile\b"]),
    ("routine", [r"\bhow (often|many times|frequently)\b", r"\busually\b"]),
]


def _infer_temporal_intent(question: str) -> str:
    """
    Classify the temporal query intent into one of:
      'before' | 'after' | 'between' | 'first' | 'last' | 'state' | 'routine' | 'lookup'

    The first pattern match wins; 'lookup' is the catch-all default.
    """
    q = str(question).lower()
    for intent, patterns in _TEMPORAL_INTENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, q):
                return intent
    return "lookup"


def _window_label_terms(window: dict[str, Any]) -> set[str]:
    """Union of content words (≥3 chars) from all action labels in the window."""
    labels = [str(x) for x in (window.get("action_labels") or [])]
    meta = window.get("metadata") or {}
    blob = " ".join(
        labels
        + [str(meta.get("start_action_label", "")), str(meta.get("end_action_label", ""))]
    )
    return _description_terms(blob)


def _score_semantic(window: dict[str, Any], query_terms: set[str]) -> float:
    """
    R_semantic = max_{a in A(w)} |query_terms ∩ terms(a)| / |query_terms|

    Takes the best-matching single action so a tight, highly relevant action
    dominates rather than being diluted by many irrelevant ones.
    """
    if not query_terms:
        return 0.0
    labels = [str(x) for x in (window.get("action_labels") or [])]
    meta = window.get("metadata") or {}
    all_labels = labels + [
        str(meta.get("start_action_label", "")),
        str(meta.get("end_action_label", "")),
    ]
    best = 0
    for lbl in all_labels:
        overlap = len(query_terms & _description_terms(lbl))
        if overlap > best:
            best = overlap
    return best / len(query_terms)


def _score_temporal(
    window: dict[str, Any],
    temporal_intent: str,
    video_duration: float | None,
) -> float:
    """
    R_temporal — compatibility of the window's position with the query's temporal intent.

    Intent → scoring rule (all in [0, 1]):
      before  → prefer windows that end early  (1 − w_end / T)
      after   → prefer windows that start late (w_start / T)
      first   → prefer windows that start earliest (1 − w_start / T)
      last    → prefer windows that start latest   (w_start / T)
      between → prefer wider windows (duration / T)
      state   → prefer longer clips  (duration / T)
      routine / lookup → neutral (0.5)
    """
    try:
        w_start = float(window.get("start", 0.0))
        w_end = float(window.get("end", w_start))
    except (TypeError, ValueError):
        return 0.5
    dur = max(0.0, w_end - w_start)
    T = max(float(video_duration), 1.0) if video_duration else 1.0

    if temporal_intent == "before":
        return 1.0 - w_end / T
    if temporal_intent == "after":
        return w_start / T
    if temporal_intent == "first":
        return 1.0 - w_start / T
    if temporal_intent == "last":
        return w_start / T
    if temporal_intent in ("between", "state"):
        return dur / T
    return 0.5  # routine / lookup


def _score_coverage(window: dict[str, Any], query_terms: set[str]) -> float:
    """
    R_coverage = |{qt in query_terms : qt appears in any action label of w}| / |query_terms|

    Fraction of distinct query concepts covered by the window, regardless of
    which individual action carries them.
    """
    if not query_terms:
        return 0.0
    return len(query_terms & _window_label_terms(window)) / len(query_terms)


def _score_cost(window: dict[str, Any], video_duration: float | None) -> float:
    """
    R_cost = duration(w) / T

    Normalised window duration; larger windows are penalised because they are
    more expensive and less discriminative.
    """
    try:
        w_start = float(window.get("start", 0.0))
        w_end = float(window.get("end", w_start))
    except (TypeError, ValueError):
        return 0.0
    T = max(float(video_duration), 1.0) if video_duration else 1.0
    return max(0.0, w_end - w_start) / T


def _score_feedback(window: dict[str, Any], feedback_terms: set[str] | None) -> float:
    """
    R_feedback = |feedback_terms ∩ label_terms(w)| / |feedback_terms|

    After a failed confidence round the LLM names what evidence was missing.
    This score boosts windows whose action labels contain those missing concepts,
    making the search adaptive to prior negative evidence.
    """
    if not feedback_terms:
        return 0.0
    return len(feedback_terms & _window_label_terms(window)) / len(feedback_terms)


def _score_action_confidence(
    window: dict[str, Any],
    confidence_timeline: list[tuple[str, str, float, float, float]],
) -> float:
    """
    Overlap-weighted mean detection confidence of predicted actions that intersect
    the window.

        R_confidence = Σ_{a ∩ w ≠ ∅} conf(a) · |a ∩ w|
                     / Σ_{a ∩ w ≠ ∅} |a ∩ w|

    Returns 0.5 (neutral) when no overlapping actions are found so the term
    does not distort scores when all windows are equally unobserved.
    """
    if not confidence_timeline:
        return 0.5
    try:
        w_start = float(window.get("start", 0.0))
        w_end = float(window.get("end", w_start))
    except (TypeError, ValueError):
        return 0.5
    if w_end <= w_start:
        return 0.5

    total_weight = 0.0
    weighted_conf = 0.0
    for _code, _name, a_start, a_end, conf in confidence_timeline:
        overlap = max(0.0, min(w_end, float(a_end)) - max(w_start, float(a_start)))
        if overlap > 0.0:
            total_weight += overlap
            weighted_conf += float(conf) * overlap
    if total_weight <= 0.0:
        return 0.5
    return weighted_conf / total_weight


def _compute_window_relevance_score(
    window: dict[str, Any],
    *,
    query_terms: set[str],
    temporal_intent: str,
    video_duration: float | None,
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    feedback_terms: set[str] | None = None,
    lam5: float = 0.0,
    confidence_timeline: list[tuple[str, str, float, float, float]] | None = None,
    lam6: float = 0.0,
) -> dict[str, Any]:
    """
    Composite relevance score:

        R(w|q) = λ1·R_semantic + λ2·R_temporal + λ3·R_coverage
               − λ4·R_cost     + λ5·R_feedback  + λ6·R_confidence

    R_feedback / λ5 are only non-zero after ≥1 failed confidence round.
    R_confidence / λ6 uses the overlap-weighted mean detection confidence from
    the pkl predictions; requires a non-empty confidence_timeline.
    """
    r_sem = _score_semantic(window, query_terms)
    r_tmp = _score_temporal(window, temporal_intent, video_duration)
    r_cov = _score_coverage(window, query_terms)
    r_cst = _score_cost(window, video_duration)
    r_fdb = _score_feedback(window, feedback_terms)
    r_conf = _score_action_confidence(window, confidence_timeline) if confidence_timeline else 0.5
    composite = (
        lam1 * r_sem + lam2 * r_tmp + lam3 * r_cov
        - lam4 * r_cst + lam5 * r_fdb + lam6 * r_conf
    )
    return {
        "score": round(composite, 6),
        "r_semantic": round(r_sem, 4),
        "r_temporal": round(r_tmp, 4),
        "r_coverage": round(r_cov, 4),
        "r_cost": round(r_cst, 4),
        "r_feedback": round(r_fdb, 4),
        "r_confidence": round(r_conf, 4),
        "temporal_intent": temporal_intent,
        "lambdas": {
            "lam1_semantic": lam1,
            "lam2_temporal": lam2,
            "lam3_coverage": lam3,
            "lam4_cost": lam4,
            "lam5_feedback": lam5,
            "lam6_confidence": lam6,
        },
    }


def _score_and_sort_windows(
    windows: list[dict[str, Any]],
    *,
    query_terms: set[str],
    temporal_intent: str,
    video_duration: float | None,
    lam1: float,
    lam2: float,
    lam3: float,
    lam4: float,
    feedback_terms: set[str] | None = None,
    lam5: float = 0.0,
    confidence_timeline: list[tuple[str, str, float, float, float]] | None = None,
    lam6: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Score every window, annotate each with ``metadata["relevance_score"]``,
    and return the list sorted descending by composite score
    (ties broken by the original v10 relevance_rank).

    Each window dict is shallow-copied so the originals are not mutated.
    When ``confidence_timeline`` is supplied and ``lam6 > 0``, windows whose
    constituent predicted actions have higher mean detection confidence are
    preferred.
    """
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for w in windows:
        scores = _compute_window_relevance_score(
            w,
            query_terms=query_terms,
            temporal_intent=temporal_intent,
            video_duration=video_duration,
            lam1=lam1, lam2=lam2, lam3=lam3, lam4=lam4,
            feedback_terms=feedback_terms,
            lam5=lam5,
            confidence_timeline=confidence_timeline,
            lam6=lam6,
        )
        w_copy = dict(w)
        meta = dict(w_copy.get("metadata") or {})
        meta["relevance_score"] = scores
        w_copy["metadata"] = meta
        prior_rank = int(meta.get("relevance_rank", 10**6))
        scored.append((scores["score"], -prior_rank, w_copy))

    scored.sort(key=lambda x: x[:2], reverse=True)
    return [w for _, _, w in scored]


def _contains_before_relation(question: str) -> bool:
    q = f" {str(question).lower()} "
    return " before " in q


def _contains_after_relation(question: str) -> bool:
    q = f" {str(question).lower()} "
    return " after " in q


def _before_anchor_query_terms(question: str) -> list[str]:
    q = str(question).strip().lower()
    if "before" not in q:
        return []
    # Prefer terms from the clause after "before ...", because that is typically the anchor action.
    after = q.split("before", 1)[1]
    after = re.split(r"[?.!,;]", after, maxsplit=1)[0]
    terms = extract_verb_terms(after) + extract_noun_terms(after)
    seen: set[str] = set()
    ordered: list[str] = []
    for t in terms:
        tt = str(t).strip().lower()
        if not tt or tt in seen:
            continue
        seen.add(tt)
        ordered.append(tt)
    return ordered


def _after_anchor_query_terms(question: str) -> list[str]:
    q = str(question).strip().lower()
    if "after" not in q:
        return []
    # Prefer terms from the clause after "after ...", because that is typically the anchor action.
    after = q.split("after", 1)[1]
    after = re.split(r"[?.!,;]", after, maxsplit=1)[0]
    terms = extract_verb_terms(after) + extract_noun_terms(after)
    seen: set[str] = set()
    ordered: list[str] = []
    for t in terms:
        tt = str(t).strip().lower()
        if not tt or tt in seen:
            continue
        seen.add(tt)
        ordered.append(tt)
    return ordered


def _reassign_relevance_ranks(windows: list[dict[str, Any]]) -> None:
    for i, w in enumerate(windows, start=1):
        meta = w.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            w["metadata"] = meta
        meta["relevance_rank"] = i


def _prioritize_before_context_window(
    ordered_windows: list[dict[str, Any]],
    *,
    question: str,
    video_actions_timeline: list[tuple[str, str, float, float]],
) -> tuple[list[dict[str, Any]], bool]:
    """
    For "before ..." questions, prioritize a window spanning:
    previous_action.start -> anchor_action.end.
    """
    if not ordered_windows or not video_actions_timeline or not _contains_before_relation(question):
        return ordered_windows, False

    terms = _before_anchor_query_terms(question)
    if not terms:
        return ordered_windows, False

    scored: list[tuple[int, int]] = []
    for idx, (_code, name, _s, _e) in enumerate(video_actions_timeline):
        lower = str(name).lower()
        score = sum(1 for t in terms if t in lower)
        scored.append((score, idx))
    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, target_idx = scored[0]
    if best_score <= 0 or target_idx <= 0:
        return ordered_windows, False

    prev_idx = target_idx - 1
    _pc, prev_name, prev_start, _prev_end = video_actions_timeline[prev_idx]
    _tc, target_name, _target_start, target_end = video_actions_timeline[target_idx]
    desired_start = float(prev_start)
    desired_end = float(target_end)
    tol = 0.05

    found_idx = -1
    for i, w in enumerate(ordered_windows):
        try:
            ws = float(w.get("start"))
            we = float(w.get("end"))
        except (TypeError, ValueError):
            continue
        if abs(ws - desired_start) > tol or abs(we - desired_end) > tol:
            continue
        inds = w.get("action_indices") or []
        if prev_idx in inds and target_idx in inds:
            found_idx = i
            break

    out = list(ordered_windows)
    if found_idx >= 0:
        chosen = dict(out.pop(found_idx))
        meta = dict(chosen.get("metadata") or {})
        meta["before_query_boost"] = True
        meta["before_anchor_action_label"] = target_name
        meta["before_previous_action_label"] = prev_name
        chosen["metadata"] = meta
        out.insert(0, chosen)
        _reassign_relevance_ranks(out)
        return out, True

    injected = {
        "start": desired_start,
        "end": desired_end,
        "action_indices": [prev_idx, target_idx],
        "action_labels": [prev_name, target_name],
        "window_type": "before_anchor_pair",
        "duration": max(0.0, desired_end - desired_start),
        "metadata": {
            "num_actions": 2,
            "source_query": question,
            "start_action_label": prev_name,
            "end_action_label": target_name,
            "query_lexical_score": best_score,
            "before_query_boost": True,
            "before_anchor_action_label": target_name,
            "before_previous_action_label": prev_name,
        },
    }
    out.insert(0, injected)
    _reassign_relevance_ranks(out)
    return out, True


def _prioritize_after_context_window(
    ordered_windows: list[dict[str, Any]],
    *,
    question: str,
    video_actions_timeline: list[tuple[str, str, float, float]],
) -> tuple[list[dict[str, Any]], bool]:
    """
    For "after ..." questions, prioritize a window spanning:
    anchor_action.start -> next_action.end.
    """
    if not ordered_windows or not video_actions_timeline or not _contains_after_relation(question):
        return ordered_windows, False

    terms = _after_anchor_query_terms(question)
    if not terms:
        return ordered_windows, False

    scored: list[tuple[int, int]] = []
    for idx, (_code, name, _s, _e) in enumerate(video_actions_timeline):
        lower = str(name).lower()
        score = sum(1 for t in terms if t in lower)
        scored.append((score, idx))
    scored.sort(key=lambda x: (-x[0], x[1]))
    best_score, anchor_idx = scored[0]
    if best_score <= 0 or anchor_idx >= (len(video_actions_timeline) - 1):
        return ordered_windows, False

    next_idx = anchor_idx + 1
    _ac, anchor_name, anchor_start, _anchor_end = video_actions_timeline[anchor_idx]
    _nc, next_name, _next_start, next_end = video_actions_timeline[next_idx]
    desired_start = float(anchor_start)
    desired_end = float(next_end)
    tol = 0.05

    found_idx = -1
    for i, w in enumerate(ordered_windows):
        try:
            ws = float(w.get("start"))
            we = float(w.get("end"))
        except (TypeError, ValueError):
            continue
        if abs(ws - desired_start) > tol or abs(we - desired_end) > tol:
            continue
        inds = w.get("action_indices") or []
        if anchor_idx in inds and next_idx in inds:
            found_idx = i
            break

    out = list(ordered_windows)
    if found_idx >= 0:
        chosen = dict(out.pop(found_idx))
        meta = dict(chosen.get("metadata") or {})
        meta["after_query_boost"] = True
        meta["after_anchor_action_label"] = anchor_name
        meta["after_next_action_label"] = next_name
        chosen["metadata"] = meta
        out.insert(0, chosen)
        _reassign_relevance_ranks(out)
        return out, True

    injected = {
        "start": desired_start,
        "end": desired_end,
        "action_indices": [anchor_idx, next_idx],
        "action_labels": [anchor_name, next_name],
        "window_type": "after_anchor_pair",
        "duration": max(0.0, desired_end - desired_start),
        "metadata": {
            "num_actions": 2,
            "source_query": question,
            "start_action_label": anchor_name,
            "end_action_label": next_name,
            "query_lexical_score": best_score,
            "after_query_boost": True,
            "after_anchor_action_label": anchor_name,
            "after_next_action_label": next_name,
        },
    }
    out.insert(0, injected)
    _reassign_relevance_ranks(out)
    return out, True


def _build_confidence_user_message_with_action_confidence(
    question: str,
    evidence_description: str,
    video_id: str,
    video_duration: float | None,
    window: dict[str, Any] | None,
    confidence_timeline: list[tuple[str, str, float, float, float]],
) -> str:
    """
    Wrapper around ``build_confidence_user_message`` that appends a structured
    summary of the predicted actions overlapping the current temporal window
    together with their model confidence scores.

    This lets the LLM weight evidence quality when deciding whether the clip
    descriptions are sufficient to answer the question — high-confidence
    detections are more trustworthy anchors than low-confidence ones.
    """
    base_msg = build_confidence_user_message(
        question=question,
        evidence_description=evidence_description,
        video_id=video_id,
        video_duration=video_duration,
    )
    if not confidence_timeline or window is None:
        return base_msg

    try:
        w_start = float(window.get("start", window.get("start_time", 0.0)))
        w_end = float(window.get("end", window.get("end_time", w_start)))
    except (TypeError, ValueError):
        return base_msg

    overlapping: list[tuple[float, str, float, float, float]] = []
    for _code, name, a_start, a_end, conf in confidence_timeline:
        overlap = max(0.0, min(w_end, float(a_end)) - max(w_start, float(a_start)))
        if overlap > 0.0:
            overlapping.append((overlap, str(name), float(a_start), float(a_end), float(conf)))
    if not overlapping:
        return base_msg

    overlapping.sort(key=lambda x: -x[0])
    lines: list[str] = [
        "",
        "[Predicted actions in this temporal window with detection confidence]",
        "The following actions were detected by the recognition model in the window "
        f"[{w_start:.1f}s – {w_end:.1f}s]. Use their confidence scores as a guide: "
        "high-confidence actions are more likely to be genuinely occurring; "
        "low-confidence actions may be borderline detections.",
    ]
    for _ov, name, a_start, a_end, conf in overlapping:
        if conf >= 0.75:
            conf_label = "high"
        elif conf >= 0.60:
            conf_label = "medium-high"
        elif conf >= 0.50:
            conf_label = "medium"
        else:
            conf_label = "low"
        lines.append(
            f"  • {name} [{a_start:.1f}s – {a_end:.1f}s]  "
            f"confidence {conf:.2f} ({conf_label})"
        )
    return base_msg + "\n".join(lines)


def _run_single_segment_round(
    *,
    question_id: str,
    question: str,
    video_id: str,
    gt_answer: Any,
    video_path: Path,
    segment: dict[str, Any],
    round_tag: str,
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    mm_infer_fn: Any,
    model_path: str,
    log: logging.Logger,
    video_actions_timeline: list[tuple[str, str, float, float]],
    video_duration: float | None,
    gen_common: dict[str, Any],
) -> dict[str, Any]:
    """Extract one segment clip, generate description payload, and return valid descriptions."""
    pred: dict[str, Any] = {
        "lookup_rank": 1,
        "code": "temporal_proposal",
        "name": "Temporal proposal segment",
        "temporal_proposal": segment,
        "round_tag": round_tag,
    }
    start_sec = segment.get("start_time")
    end_sec = segment.get("end_time")
    try:
        start_sec_f = float(start_sec) if start_sec is not None else None
        end_sec_f = float(end_sec) if end_sec is not None else None
    except (TypeError, ValueError):
        start_sec_f, end_sec_f = None, None

    if start_sec_f is None or end_sec_f is None or end_sec_f < start_sec_f:
        pred["error"] = "invalid_temporal_proposal_segment"
        return {
            "prediction": pred,
            "texts": [],
            "seg_keys": [],
            "desc_payload": None,
            "desc_json_path": None,
            "segment_dir": None,
        }

    pred["start_sec"] = start_sec_f
    pred["end_sec"] = end_sec_f
    seg_dir = args.video_seg_root / f"{video_id}_{question_id}_{round_tag}"
    desc_json_path = args.s2_out / f"segment_descriptions_{question_id}_{round_tag}.json"

    mp4s = list(seg_dir.glob("*.mp4"))
    if not (args.reuse_intermediate and mp4s):
        try:
            run_extract_gt_timeline(
                video_path,
                video_id,
                [("proposal", "Temporal proposal segment", start_sec_f, end_sec_f)],
                seg_dir,
                context_sec=args.extract_context_sec,
                min_clip_sec=args.extract_min_clip_sec,
            )
        except RuntimeError as e:
            log.error("%s [%s]: %s", question_id, round_tag, e)
            pred["error"] = str(e)
            mp4s = []
        else:
            mp4s = list(seg_dir.glob("*.mp4"))

    if not mp4s and "error" not in pred:
        pred["error"] = "no_segments"
        pred["segment_dir"] = str(seg_dir.resolve())

    desc_payload: dict[str, Any] | None = None
    if "error" not in pred:
        if args.reuse_intermediate and desc_json_path.is_file():
            try:
                desc_payload = json.loads(desc_json_path.read_text(encoding="utf-8"))
                log.info("Reuse: descriptions %s [%s] from %s", question_id, round_tag, desc_json_path)
            except Exception:
                desc_payload = None

        if desc_payload is None:
            raw_justification: str = str(segment.get("justification") or "").strip()
            if ": " in raw_justification:
                justification_hint: str | None = raw_justification.split(": ", 1)[1].strip() or None
            else:
                justification_hint = raw_justification or None

            vlm_hint: str | None = None
            llm_initial_raw = ""
            hint_source = "justification_fallback"
            if (
                not args.skip_llm_timeline_initial_answer
                and video_actions_timeline
            ):
                try:
                    vlm_hint, llm_initial_raw = run_llm_timeline_initial_answer(
                        question=question,
                        video_id=video_id,
                        video_duration=video_duration,
                        w_start=start_sec_f,
                        w_end=end_sec_f,
                        video_actions_timeline=video_actions_timeline,
                        model=model,
                        processor=processor,
                        mm_infer_fn=mm_infer_fn,
                        gen_common=gen_common,
                        max_new_tokens=int(args.llm_initial_answer_max_new_tokens),
                        global_head_max=int(args.llm_initial_answer_global_head_actions),
                    )
                    if vlm_hint:
                        hint_source = "llm_timeline_initial_answer"
                except Exception as _init_exc:
                    log.warning(
                        "%s [%s]: LLM timeline initial answer failed: %s",
                        question_id,
                        round_tag,
                        _init_exc,
                    )
                    vlm_hint = None
                    llm_initial_raw = ""

            if not (vlm_hint and str(vlm_hint).strip()):
                vlm_hint = justification_hint
                hint_source = "justification_fallback"

            pred["vlm_hint_source"] = hint_source
            pred["llm_timeline_initial_answer_raw"] = llm_initial_raw or None

            desc_payload = build_segment_descriptions_payload(
                question_id,
                seg_dir,
                model,
                processor,
                mm_infer_fn,
                args.desc_fps,
                args.desc_max_frames,
                args.desc_max_new_tokens,
                model_path,
                args.desc_glob,
                "_",
                question,
                args.desc_prompt_style,
                action_context=vlm_hint,
            )
            desc_payload["ground_truth_answer"] = gt_answer
            desc_payload["action"] = {
                "code": "temporal_proposal",
                "name": "Temporal proposal segment",
                "start_sec": start_sec_f,
                "end_sec": end_sec_f,
                "justification": raw_justification,
                "justification_hint_fallback": justification_hint,
                "action_context_hint": vlm_hint,
                "vlm_hint_source": hint_source,
                "vlm_hint_text": vlm_hint,
                "llm_timeline_initial_answer_raw": llm_initial_raw or None,
            }
            desc_json_path.write_text(
                json.dumps(desc_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    texts: list[str] = []
    seg_keys: list[str] = []
    if desc_payload is not None:
        texts, seg_keys = _collect_valid_descriptions(desc_payload)
        if not texts:
            pred["error"] = "no_valid_descriptions"
            pred["descriptions_json"] = str(desc_json_path.resolve())
        else:
            pred["descriptions_json"] = str(desc_json_path.resolve())
            pred["segment_keys"] = seg_keys
    return {
        "prediction": pred,
        "texts": texts,
        "seg_keys": seg_keys,
        "desc_payload": desc_payload,
        "desc_json_path": desc_json_path,
        "segment_dir": seg_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA,
        help="OTB JSON file (single top-level object with question_id keys).",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=_DEFAULT_CLASSES,
        help="OTB action-list file (format: index<TAB>name) used to map pkl class indices to names.",
    )
    parser.add_argument(
        "--otb-json",
        type=Path,
        default=_DEFAULT_OTB_JSON,
        help="OTB annotation JSON (default: data/smarthome.json; optional, kept for compatibility).",
    )
    parser.add_argument(
        "--otb-video-root",
        type=Path,
        default=_DEFAULT_OTB_VIDEO_ROOT,
        help="Directory containing <video_id>.mp4 (OTB).",
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=_DEFAULT_PKL,
        help="Pickle file (video_id -> logits [num_classes, T]) with per-frame AD predictions.",
    )
    parser.add_argument(
        "--video-seg-root",
        type=Path,
        default=_DEFAULT_VIDEO_SEG_ROOT,
        help="Parent folder for per-question per-action segment directories.",
    )
    parser.add_argument(
        "--s2-out",
        type=Path,
        default=_DEFAULT_SEG_PARENT,
        help="Writes llm_actions_* and segment_descriptions_* files here.",
    )
    parser.add_argument(
        "--final-out-dir",
        type=Path,
        default=_DEFAULT_FINAL_DIR,
        help="Per-question final JSON output directory.",
    )
    parser.add_argument("--model-path", default=None, help="HF id or local checkpoint.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visual-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens-answer", type=int, default=2048)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--max-new-tokens-semantic",
        type=int,
        default=2048,
        help="Compatibility flag from older subset pipeline; unused.",
    )
    parser.add_argument(
        "--max-new-tokens-first-pass",
        type=int,
        default=None,
        help="Compatibility flag from older subset pipeline; unused.",
    )
    parser.add_argument(
        "--skip-first-pass",
        action="store_true",
        help="Compatibility flag from older subset pipeline; unused.",
    )
    parser.add_argument(
        "--extract-threshold",
        type=float,
        default=0.50,
        help=(
            "Raw probability threshold applied to pkl frame-level scores (no sigmoid). "
            "The pkl stores probabilities already in [0, 1]; frames where "
            "prob >= threshold are treated as active for that class. "
            "Default 0.50 (recommended; empirically matches GT segment count best)."
        ),
    )
    parser.add_argument(
        "--pool-factor",
        type=int,
        default=16,
        help=(
            "Temporal pooling window used when the AD model was trained/evaluated. "
            "pkl_frame_index × pool_factor = actual video frame number. "
            "Default 16 (verified for the AD pkl: pkl_T × 16 ≈ video nb_frames)."
        ),
    )
    parser.add_argument(
        "--merge-gap-frames",
        type=int,
        default=1,
        help=(
            "Merge same-class predicted segments separated by at most this many pooled frames. "
            "0 = merge only touching segments; 1 (default) = allow one uncovered frame; "
            "-1 = disable merging."
        ),
    )
    parser.add_argument(
        "--skip-llm-query-timeline-merge",
        action="store_true",
        help=(
            "Use v10's default windows (includes neighbor-chain merged spans) instead of one "
            "query-guided LLM pass to decide which timeline actions to merge for evidence."
        ),
    )
    parser.add_argument(
        "--llm-merge-timeline-max-new-tokens",
        type=int,
        default=384,
        help="Max new tokens for the per-question JSON timeline-merge LLM call (text-only).",
    )
    parser.add_argument(
        "--llm-merge-timeline-max-actions",
        type=int,
        default=48,
        help=(
            "Max actions listed in the merge prompt (chronological head). "
            "0 = list the full timeline (can be slow/large for long videos)."
        ),
    )
    parser.add_argument(
        "--skip-llm-timeline-initial-answer",
        action="store_true",
        help=(
            "Skip the text-only LLM that infers a tentative answer from predicted action labels "
            "and times before the VLM description pass; use the segment justification snippet "
            "as the clip hint instead (legacy behaviour)."
        ),
    )
    parser.add_argument(
        "--llm-initial-answer-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for the text-only tentative answer from actions + timestamps (per clip window).",
    )
    parser.add_argument(
        "--llm-initial-answer-global-head-actions",
        type=int,
        default=48,
        help=(
            "When no predicted actions overlap the clip window, include this many earliest "
            "timeline entries in the LLM prompt for global context."
        ),
    )
    parser.add_argument(
        "--no-fallback-max-segment",
        action="store_true",
        help="Compatibility flag from older pipeline; unused.",
    )
    parser.add_argument(
        "--extract-bin-context-pad",
        type=int,
        default=1,
        help="Compatibility flag from older pipeline; unused.",
    )
    parser.add_argument("--desc-fps", type=int, default=1)
    parser.add_argument("--desc-max-frames", type=int, default=180)
    parser.add_argument("--desc-max-new-tokens", type=int, default=512)
    parser.add_argument("--segment-max-new-tokens", type=int, default=256)
    parser.add_argument("--confidence-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--score-lam-semantic",
        type=float,
        default=0.50,
        help="λ1: weight for R_semantic (action–query lexical alignment). Default 0.50.",
    )
    parser.add_argument(
        "--score-lam-temporal",
        type=float,
        default=0.20,
        help="λ2: weight for R_temporal (window position vs temporal intent). Default 0.20.",
    )
    parser.add_argument(
        "--score-lam-coverage",
        type=float,
        default=0.20,
        help="λ3: weight for R_coverage (fraction of query terms covered). Default 0.20.",
    )
    parser.add_argument(
        "--score-lam-cost",
        type=float,
        default=0.10,
        help="λ4: penalty weight for R_cost (normalised window duration). Default 0.10.",
    )
    parser.add_argument(
        "--score-lam-feedback",
        type=float,
        default=0.60,
        help=(
            "λ5: weight for R_feedback (match to LLM-diagnosed missing evidence). "
            "Only active after ≥1 failed confidence round. Default 0.60."
        ),
    )
    parser.add_argument(
        "--score-lam-confidence",
        type=float,
        default=0.20,
        help=(
            "λ6: weight for R_confidence (overlap-weighted mean detection confidence "
            "of predicted actions in the window, derived from the pkl scores). "
            "Set to 0 to disable. Default 0.20."
        ),
    )
    parser.add_argument("--max-segment-refinements", type=int, default=1)
    parser.add_argument(
        "--overlap-stop-iou",
        type=float,
        default=_STOP_OVERLAP_IOU,
        help="Stop refinement when IoU(next_segment, previous_segment) exceeds this value.",
    )
    parser.add_argument(
        "--skip-confidence-refine",
        action="store_true",
        help="Use only the single highest-relevance v10 window (one clip + QA); skip confidence model and multi-window iteration.",
    )
    parser.add_argument(
        "--max-temporal-windows",
        type=int,
        default=0,
        help="Max v10 windows to try in order (0 = no cap). Each window: clip -> describe -> confidence -> QA if confident.",
    )
    parser.add_argument("--desc-glob", type=str, default="*.mp4")
    parser.add_argument(
        "--desc-prompt-style",
        type=str,
        default="query_soft",
        choices=("generic", "query_soft", "query_strict"),
        help="Prompt style used for segment description generation.",
    )
    parser.add_argument(
        "--extract-context-sec",
        type=float,
        default=0.0,
        help="Pad each GT action clip by this many seconds on both sides.",
    )
    parser.add_argument(
        "--extract-min-clip-sec",
        type=float,
        default=0.0,
        help="Ensure each extracted action clip is at least this long (seconds).",
    )
    parser.add_argument("--top-k-actions", type=int, default=10, help="Top-K ordered actions for QA + voting.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a question if final answer JSON already exists.",
    )
    parser.add_argument(
        "--question-ids",
        type=str,
        nargs="*",
        default=None,
        help="If set, only run these question ids (keys in --data).",
    )
    parser.add_argument(
        "--answer-instruction",
        default="",
        help="Override final QA system prompt.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel shards (processes).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index for this process in [0, num_shards).",
    )
    parser.add_argument(
        "--reuse-intermediate",
        action="store_true",
        help="Reuse existing llm_actions / segment_descriptions / segment clips when available.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("pipeline_unified_v10_tempseg")

    if args.top_k_actions <= 0:
        log.error("--top-k-actions must be >= 1, got %s", args.top_k_actions)
        return 1
    if args.extract_context_sec < 0:
        log.error("--extract-context-sec must be >= 0, got %s", args.extract_context_sec)
        return 1
    if args.extract_min_clip_sec < 0:
        log.error("--extract-min-clip-sec must be >= 0, got %s", args.extract_min_clip_sec)
        return 1
    if args.overlap_stop_iou < 0.0 or args.overlap_stop_iou > 1.0:
        log.error("--overlap-stop-iou must be in [0, 1], got %s", args.overlap_stop_iou)
        return 1
    if args.max_temporal_windows < 0:
        log.error("--max-temporal-windows must be >= 0, got %s", args.max_temporal_windows)
        return 1
    if not args.data.is_file():
        log.error("Data file not found: %s", args.data)
        return 1
    if not args.classes_file.is_file():
        log.error("Classes file not found: %s", args.classes_file)
        return 1
    if not args.pkl.is_file():
        log.error("Pkl predictions file not found: %s", args.pkl)
        return 1
    if args.num_shards <= 0:
        log.error("--num-shards must be >= 1, got %s", args.num_shards)
        return 1
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        log.error(
            "--shard-index must be in [0, %s), got %s",
            args.num_shards,
            args.shard_index,
        )
        return 1

    raw = args.data.read_text(encoding="utf-8").strip()
    data: dict[str, Any] = json.loads(raw)
    otb_classes = load_otb_classes(args.classes_file)
    if not otb_classes:
        log.error("No classes loaded from %s", args.classes_file)
        return 1
    # Ordered list of class names indexed by class id (0 … N-1).
    otb_class_names: list[str] = [name for _, _, name in otb_classes]

    log.info("Loading pkl predictions from %s", args.pkl)
    with open(args.pkl, "rb") as _pkl_f:
        pkl_data: dict = pickle.load(_pkl_f)
    if not isinstance(pkl_data, dict):
        log.error("pkl file should contain a dict keyed by video_id, got %s", type(pkl_data).__name__)
        return 1
    log.info(
        "Loaded pkl predictions for %d video(s); threshold=%.4f  pool_factor=%d",
        len(pkl_data), args.extract_threshold, args.pool_factor,
    )

    model_path = args.model_path or "DAMO-NLP-SG/VideoLLaMA3-7B"
    gen_common: dict[str, Any] = {"do_sample": args.do_sample}
    if args.temperature is not None:
        gen_common["temperature"] = args.temperature
    if args.top_p is not None:
        gen_common["top_p"] = args.top_p

    seed_everything(args.seed)
    disable_torch_init()
    model_init_fn, mm_infer_fn = INFERENCES(model_path)
    model, processor = model_init_fn(
        model_path,
        args.max_visual_tokens,
        device_map={"": args.device},
    )

    keys = list(data.keys())
    if args.question_ids:
        want = set(args.question_ids)
        keys = [k for k in keys if k in want]
    if args.num_shards > 1:
        keys = [k for i, k in enumerate(keys) if i % args.num_shards == args.shard_index]
        log.info(
            "Shard %d/%d will process %d question(s)",
            args.shard_index,
            args.num_shards,
            len(keys),
        )

    args.video_seg_root.mkdir(parents=True, exist_ok=True)
    args.s2_out.mkdir(parents=True, exist_ok=True)
    args.final_out_dir.mkdir(parents=True, exist_ok=True)

    qa_instruction = (
        args.answer_instruction.strip()
        if args.answer_instruction.strip()
        else "You are a helpful assistant that answers questions about a video. For the final answer step, "
        "you will see the same short clip again together with a query-conditioned written description of that clip. "
        "Use both the pixels and the description; prefer the video if they conflict. Answer directly and concisely."
    )
    gen_ans = {**gen_common, "max_new_tokens": args.max_new_tokens_answer}

    for question_id in keys:
        rec = data[question_id]
        question = rec.get("question", "")
        video_id = rec.get("video_id", "")
        gt_answer = rec.get("answer")

        final_path = args.final_out_dir / f"{question_id}.json"
        if args.resume and final_path.is_file():
            log.info("Resume: skip %s (exists %s)", question_id, final_path)
            continue
        if not question or not video_id:
            log.warning("Skip %s: missing question or video_id", question_id)
            continue

        video_path = args.otb_video_root / f"{video_id}.mp4"
        if not video_path.is_file():
            log.error("Video not found for %s: %s", question_id, video_path)
            continue
        if video_id not in pkl_data:
            log.error("%s: video_id %s not found in pkl %s", question_id, video_id, args.pkl)
            continue

        qstrip = str(question).strip()
        confidence_timeline, video_duration = get_predicted_actions_with_confidence_from_pkl(
            video_id=video_id,
            pkl_data=pkl_data,
            class_names=otb_class_names,
            threshold=args.extract_threshold,
            video_path=video_path,
            merge_gap_frames=args.merge_gap_frames,
            pool_factor=args.pool_factor,
        )
        # Strip confidence component for functions that expect 4-tuple timeline.
        video_actions_timeline: list[tuple[str, str, float, float]] = [
            (code, name, s, e) for code, name, s, e, _conf in confidence_timeline
        ]
        if not video_actions_timeline:
            log.error(
                "%s: no predicted actions above threshold %.4f for %s",
                question_id, args.extract_threshold, video_id,
            )
            continue

        noun_terms = extract_noun_terms(qstrip)
        verb_terms = extract_verb_terms(qstrip)
        actions_json_path = args.s2_out / f"llm_actions_{question_id}_ordered_actions.json"

        want_llm_timeline_merge = not args.skip_llm_query_timeline_merge
        ordered_actions: list[dict[str, Any]] = []
        ordered_windows: list[dict[str, Any]] | None = None
        llm_timeline_merge_raw = ""
        cached_stage1 = ""

        if args.reuse_intermediate and actions_json_path.is_file():
            try:
                cached = json.loads(actions_json_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict):
                    oa = (cached.get("ordered_actions") or {}).get("actions")
                    if isinstance(oa, list):
                        ordered_actions = oa
                        log.info("Reuse: ordered actions for %s from %s", question_id, actions_json_path)
                    cached_used_llm = cached.get("used_llm_timeline_merge")
                    ow = cached.get("temporal_windows_ordered_by_query")
                    if (
                        isinstance(ow, list)
                        and ow
                        and isinstance(cached_used_llm, bool)
                        and cached_used_llm == want_llm_timeline_merge
                    ):
                        ordered_windows = ow
                        llm_timeline_merge_raw = str(cached.get("llm_timeline_merge_raw") or "")
                        cached_stage1 = str(cached.get("stage1_strategy") or "")
                        log.info(
                            "Reuse: temporal windows (%s) for %s from %s",
                            "llm_merge" if want_llm_timeline_merge else "v10_default",
                            question_id,
                            actions_json_path,
                        )
            except Exception:
                ordered_actions = []
                ordered_windows = None

        if not ordered_actions:
            ordered_actions = build_query_guided_action_lookup_order(
                question=qstrip,
                video_actions_timeline=video_actions_timeline,
                noun_terms=noun_terms,
                verb_terms=verb_terms,
            )

        top_actions = ordered_actions[: args.top_k_actions]

        if ordered_windows is None:
            if args.skip_llm_query_timeline_merge:
                ordered_windows = compute_ordered_temporal_windows_for_query(
                    qstrip, video_actions_timeline, video_duration
                )
                stage1_strategy = "v10_compute_ordered_windows"
                llm_timeline_merge_raw = ""
            else:
                m_cap = (
                    len(video_actions_timeline)
                    if args.llm_merge_timeline_max_actions <= 0
                    else args.llm_merge_timeline_max_actions
                )
                ordered_windows, llm_timeline_merge_raw = compute_ordered_temporal_windows_llm_query_merge(
                    qstrip,
                    video_actions_timeline,
                    video_duration,
                    model=model,
                    processor=processor,
                    mm_infer_fn=mm_infer_fn,
                    gen_common=gen_common,
                    merge_max_new_tokens=args.llm_merge_timeline_max_new_tokens,
                    max_actions_in_prompt=m_cap,
                    video_id=video_id,
                    log=log,
                )
                stage1_strategy = "v10_llm_query_guided_merge"
        else:
            stage1_strategy = cached_stage1 or (
                "v10_llm_query_guided_merge" if want_llm_timeline_merge else "v10_compute_ordered_windows"
            )
        if not ordered_windows:
            log.error("%s: v10 produced no temporal windows", question_id)
            continue
        ordered_windows, before_boost_applied = _prioritize_before_context_window(
            ordered_windows,
            question=qstrip,
            video_actions_timeline=video_actions_timeline,
        )
        ordered_windows, after_boost_applied = _prioritize_after_context_window(
            ordered_windows,
            question=qstrip,
            video_actions_timeline=video_actions_timeline,
        )

        windows_iter: list[dict[str, Any]] = list(ordered_windows)
        if args.max_temporal_windows > 0:
            windows_iter = windows_iter[: args.max_temporal_windows]
        if args.skip_confidence_refine:
            windows_iter = windows_iter[:1]

        # ── Relevance scoring ──────────────────────────────────────────────
        # Build the unified query-term set and infer the temporal intent once
        # per question, then score+sort all candidate windows.
        query_terms: set[str] = {
            t.lower()
            for t in noun_terms + verb_terms
            if isinstance(t, str) and t.strip()
        }
        temporal_intent = _infer_temporal_intent(qstrip)
        log.info("%s: temporal_intent=%s  query_terms=%s", question_id, temporal_intent, sorted(query_terms))

        lam_kwargs: dict[str, Any] = dict(
            lam1=args.score_lam_semantic,
            lam2=args.score_lam_temporal,
            lam3=args.score_lam_coverage,
            lam4=args.score_lam_cost,
            lam6=args.score_lam_confidence,
            confidence_timeline=confidence_timeline if args.score_lam_confidence > 0.0 else None,
        )
        # Score and sort the initial window list; each window dict gets a
        # metadata["relevance_score"] annotation with all sub-scores.
        windows_iter = _score_and_sort_windows(
            windows_iter,
            query_terms=query_terms,
            temporal_intent=temporal_intent,
            video_duration=video_duration,
            **lam_kwargs,
        )

        temporal_proposal = _window_dict_to_segment(
            windows_iter[0], video_duration=video_duration
        )
        model_temporal_proposal = None
        used_temporal_fallback = False

        actions_record = {
            "question_id": question_id,
            "video_id": video_id,
            "ground_truth_answer": gt_answer,
            "input": qstrip,
            "ordered_actions": {
                "video_id": video_id,
                "video_duration_sec": video_duration,
                "noun_terms": noun_terms,
                "verb_terms": verb_terms,
                "actions": ordered_actions,
            },
            "top_k_actions_considered": args.top_k_actions,
            "actions_used_for_qa": top_actions,
            "temporal_windows_ordered_by_query": ordered_windows,
            "used_llm_timeline_merge": want_llm_timeline_merge,
            "llm_timeline_merge_raw": llm_timeline_merge_raw,
            "windows_iterated_for_clips": windows_iter,
            "max_temporal_windows_cap": args.max_temporal_windows,
            "temporal_proposal": temporal_proposal,
            "temporal_proposal_model_raw": model_temporal_proposal,
            "used_fallback_temporal_segment": used_temporal_fallback,
            "model_path": model_path,
            "stage1_strategy": stage1_strategy,
            "refinement_policy": "v10_confidence_gate_iterative_windows",
            "before_context_boost_applied": before_boost_applied,
            "after_context_boost_applied": after_boost_applied,
            "temporal_intent": temporal_intent,
            "query_terms": sorted(query_terms),
            "score_lambdas": {
                "lam1_semantic": args.score_lam_semantic,
                "lam2_temporal": args.score_lam_temporal,
                "lam3_coverage": args.score_lam_coverage,
                "lam4_cost": args.score_lam_cost,
                "lam5_feedback": args.score_lam_feedback,
                "lam6_confidence": args.score_lam_confidence,
            },
        }
        actions_json_path.write_text(
            json.dumps(actions_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        proposal_pred: dict[str, Any] = {
            "lookup_rank": 1,
            "code": "temporal_proposal",
            "name": (
                "v10 temporal windows (query LLM merge)"
                if want_llm_timeline_merge
                else "v10 temporal windows"
            ),
            "instruction": qa_instruction,
            "generation": gen_ans,
            "temporal_proposal": temporal_proposal,
            "temporal_proposal_model_raw": model_temporal_proposal,
            "temporal_windows_ordered_by_query": ordered_windows,
        }
        action_predictions: list[dict[str, Any]] = [proposal_pred]

        v10_round_traces: list[dict[str, Any]] = []
        final_answer_output: str | None = None
        final_answer_input: str | None = None
        final_answer_seg_keys: list[str] = []
        all_acc_texts: list[str] = []
        all_acc_keys: list[str] = []
        last_round_texts: list[str] = []
        last_round_keys: list[str] = []
        last_round_seg_dir: Path | None = None
        stopped_on_relevance_hit = False
        last_segment_dict: dict[str, Any] | None = None
        winning_round_tag: str | None = None
        merged_first_desc_payload = False
        description_guided_reorders = 0
        # Cumulative feedback terms grow with each failed confidence round so that
        # every piece of negative evidence is carried forward.
        cumulative_feedback_terms: set[str] = set()
        windows_queue: list[dict[str, Any]] = list(windows_iter)
        processed_windows = 0

        while windows_queue:
            wdict = windows_queue.pop(0)
            seg = _window_dict_to_segment(wdict, video_duration=video_duration)
            last_segment_dict = seg
            meta = wdict.get("metadata") or {}
            rank = int(meta.get("relevance_rank", processed_windows + 1))
            wtype = str(wdict.get("window_type", "win"))
            round_tag = f"v10_r{rank:02d}_{wtype}"

            round_result = _run_single_segment_round(
                question_id=question_id,
                question=qstrip,
                video_id=video_id,
                gt_answer=gt_answer,
                video_path=video_path,
                segment=seg,
                round_tag=round_tag,
                args=args,
                model=model,
                processor=processor,
                mm_infer_fn=mm_infer_fn,
                model_path=model_path,
                log=log,
                video_actions_timeline=video_actions_timeline,
                video_duration=video_duration,
                gen_common=gen_common,
            )
            rr = round_result["prediction"]
            trace: dict[str, Any] = {
                "window_index": processed_windows,
                "round_tag": round_tag,
                "window": wdict,
                "segment": seg,
                "relevance_score": (wdict.get("metadata") or {}).get("relevance_score"),
                "prediction": rr,
            }
            processed_windows += 1
            if rr.get("error"):
                trace["skipped_reason"] = rr["error"]
                v10_round_traces.append(trace)
                log.warning("%s: %s — %s", question_id, round_tag, rr.get("error"))
                continue

            texts = round_result["texts"]
            seg_keys_round = [f"{round_tag}:{k}" for k in round_result["seg_keys"]]
            if not texts:
                trace["skipped_reason"] = "no_description_texts"
                v10_round_traces.append(trace)
                continue

            last_round_texts = list(texts)
            last_round_keys = list(seg_keys_round)
            last_round_seg_dir = Path(round_result["segment_dir"]) if round_result.get("segment_dir") else None
            all_acc_texts, all_acc_keys = _append_unique_descriptions(
                all_acc_texts, all_acc_keys, texts, seg_keys_round
            )

            # Ask the LLM whether the descriptions are sufficient to answer the question.
            confidence_val: int | None = None
            confidence_reason: str = ""
            if not args.skip_confidence_refine:
                evidence_text = "\n\n".join(texts)
                conf_system = build_confidence_system_instruction()
                conf_user = _build_confidence_user_message_with_action_confidence(
                    question=qstrip,
                    evidence_description=evidence_text,
                    video_id=video_id,
                    video_duration=video_duration,
                    window=wdict,
                    confidence_timeline=confidence_timeline,
                )
                gen_conf = {**gen_common, "max_new_tokens": args.confidence_max_new_tokens}
                try:
                    conf_raw = run_text_only(
                        model, processor, mm_infer_fn, conf_system, conf_user, gen_conf
                    )
                    confidence_val, confidence_reason = parse_confidence_json_from_model_output(conf_raw)
                except Exception as _conf_exc:
                    log.warning("%s [%s]: confidence call failed: %s", question_id, round_tag, _conf_exc)
                    conf_raw = ""
                trace["confidence_raw"] = conf_raw if not args.skip_confidence_refine else "skipped"
                trace["confidence"] = confidence_val
                trace["confidence_reason"] = confidence_reason
                log.info(
                    "%s [%s]: confidence=%s reason=%r",
                    question_id, round_tag, confidence_val, confidence_reason,
                )
            should_stop = args.skip_confidence_refine or (confidence_val == 1)

            # After each description pass, re-score and re-sort remaining windows.
            # When confidence=0, the LLM's reason identifies missing evidence;
            # extract those terms and add them to the cumulative feedback set so
            # R_feedback boosts windows that carry the missing concepts.
            if windows_queue:
                if confidence_val == 0 and confidence_reason:
                    new_reason_terms = _description_terms(confidence_reason)
                    new_reason_terms.update(
                        t.lower()
                        for t in extract_noun_terms(confidence_reason) + extract_verb_terms(confidence_reason)
                        if isinstance(t, str) and t.strip()
                    )
                    cumulative_feedback_terms.update(new_reason_terms)
                    trace["confidence_reason_terms_used_for_rerank"] = sorted(new_reason_terms)
                    log.info(
                        "%s [%s]: confidence=0 — cumulative feedback terms: %s",
                        question_id, round_tag, sorted(cumulative_feedback_terms),
                    )

                active_feedback = cumulative_feedback_terms if cumulative_feedback_terms else None
                prev_ids = [id(w) for w in windows_queue]
                windows_queue = _score_and_sort_windows(
                    windows_queue,
                    query_terms=query_terms,
                    temporal_intent=temporal_intent,
                    video_duration=video_duration,
                    feedback_terms=active_feedback,
                    lam5=args.score_lam_feedback if active_feedback else 0.0,
                    **lam_kwargs,
                )
                reordered = any(id(a) != b for a, b in zip(windows_queue, prev_ids))
                trace["remaining_windows_reordered"] = reordered
                trace["cumulative_feedback_terms"] = sorted(cumulative_feedback_terms)
                if reordered:
                    description_guided_reorders += 1

            v10_round_traces.append(trace)

            if not merged_first_desc_payload:
                proposal_pred.update(rr)
                merged_first_desc_payload = True

            if should_stop:
                user_block = build_segment_qa_user_content(texts, qstrip, seg_keys_round)
                seg_dir_ans = Path(round_result["segment_dir"]) if round_result.get("segment_dir") else None
                mp4s_ans = sorted(seg_dir_ans.glob(args.desc_glob)) if seg_dir_ans else []
                if mp4s_ans:
                    try:
                        final_llm_out = run_vlm_final_answer_from_clip(
                            mp4s_ans[0],
                            question=qstrip,
                            description_texts=texts,
                            system_instruction=qa_instruction,
                            model=model,
                            processor=processor,
                            mm_infer_fn=mm_infer_fn,
                            fps=args.desc_fps,
                            max_frames=args.desc_max_frames,
                            gen_kwargs=gen_ans,
                        )
                    except Exception as _vqa_exc:
                        log.warning(
                            "%s [%s]: VLM final QA failed (%s); falling back to text-only QA.",
                            question_id,
                            round_tag,
                            _vqa_exc,
                        )
                        final_llm_out = run_text_only(
                            model,
                            processor,
                            mm_infer_fn,
                            qa_instruction,
                            user_block,
                            gen_ans,
                        )
                else:
                    log.warning(
                        "%s [%s]: no clip mp4 for VLM final QA; using text-only QA on descriptions.",
                        question_id,
                        round_tag,
                    )
                    final_llm_out = run_text_only(
                        model,
                        processor,
                        mm_infer_fn,
                        qa_instruction,
                        user_block,
                        gen_ans,
                    )
                final_answer_output = final_llm_out
                final_answer_input = user_block
                final_answer_seg_keys = list(seg_keys_round)
                winning_round_tag = round_tag
                stopped_on_relevance_hit = True
                break

        proposal_pred["v10_window_round_traces"] = v10_round_traces
        proposal_pred["temporal_proposal_final"] = last_segment_dict
        proposal_pred["refinement_policy"] = "llm_confidence_gate_iterative_windows"
        proposal_pred["stopped_after_confident_window"] = winning_round_tag
        proposal_pred["description_guided_window_reorders"] = description_guided_reorders

        if (
            stopped_on_relevance_hit
            and isinstance(final_answer_output, str)
            and final_answer_output.strip()
        ):
            proposal_pred["input"] = final_answer_input
            proposal_pred["output"] = final_answer_output
            proposal_pred["segment_keys"] = final_answer_seg_keys
            proposal_pred["used_last_description_fallback"] = False
        elif last_round_texts:
            proposal_pred["used_last_description_fallback"] = True
            user_block = build_segment_qa_user_content(last_round_texts, qstrip, last_round_keys)
            mp4s_fb = sorted(last_round_seg_dir.glob(args.desc_glob)) if last_round_seg_dir else []
            if mp4s_fb:
                try:
                    final_llm_out = run_vlm_final_answer_from_clip(
                        mp4s_fb[0],
                        question=qstrip,
                        description_texts=last_round_texts,
                        system_instruction=qa_instruction,
                        model=model,
                        processor=processor,
                        mm_infer_fn=mm_infer_fn,
                        fps=args.desc_fps,
                        max_frames=args.desc_max_frames,
                        gen_kwargs=gen_ans,
                    )
                except Exception as _vqa_exc_fb:
                    log.warning(
                        "%s: VLM final QA (fallback) failed (%s); using text-only QA.",
                        question_id,
                        _vqa_exc_fb,
                    )
                    final_llm_out = run_text_only(
                        model,
                        processor,
                        mm_infer_fn,
                        qa_instruction,
                        user_block,
                        gen_ans,
                    )
            else:
                log.warning(
                    "%s: no clip path for VLM final QA fallback; using text-only QA.",
                    question_id,
                )
                final_llm_out = run_text_only(
                    model,
                    processor,
                    mm_infer_fn,
                    qa_instruction,
                    user_block,
                    gen_ans,
                )
            proposal_pred["input"] = user_block
            proposal_pred["output"] = final_llm_out
            proposal_pred["segment_keys"] = last_round_keys
            proposal_pred["fallback_answer_source"] = "last_segment_description"
        elif "error" not in proposal_pred:
            proposal_pred["error"] = "no_segment_description_for_final_answer"

        direct_final_answer = proposal_pred.get("output")
        if not isinstance(direct_final_answer, str) or not direct_final_answer.strip():
            direct_final_answer = None
        final_record = {
            "query": qstrip,
            "model_path": model_path,
            "question_id": question_id,
            "video_id": video_id,
            "ground_truth_answer": gt_answer,
            "ordered_actions_metadata": actions_record["ordered_actions"],
            "temporal_proposal": temporal_proposal,
            "temporal_proposal_model_raw": None,
            "temporal_windows_ordered_by_query": ordered_windows,
            "temporal_proposal_final": proposal_pred.get("temporal_proposal_final"),
            "top_k_actions": args.top_k_actions,
            "actions_used_for_qa": action_predictions,
            "action_predictions": action_predictions,
            "decision_method": "v10_ordered_windows_llm_confidence_stop",
            "temporal_intent": temporal_intent,
            "query_terms": sorted(query_terms),
            "score_lambdas": {
                "lam1_semantic": args.score_lam_semantic,
                "lam2_temporal": args.score_lam_temporal,
                "lam3_coverage": args.score_lam_coverage,
                "lam4_cost": args.score_lam_cost,
                "lam5_feedback": args.score_lam_feedback,
                "lam6_confidence": args.score_lam_confidence,
            },
            "majority_vote": None,
            "final_answer": direct_final_answer,
            "actions_json": str(actions_json_path.resolve()),
        }
        final_path.write_text(
            json.dumps(final_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log.info(
            "Wrote %s (final=%r)",
            final_path,
            direct_final_answer,
        )

    del model
    try:
        torch_mod = __import__("torch")
        torch_mod.cuda.empty_cache()
    except Exception:
        pass
    if args.num_shards == 1:
        cleanup_s2_intermediates_if_complete(
            args.data,
            args.final_out_dir,
            args.s2_out,
            log=log,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
