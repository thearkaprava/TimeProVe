#!/usr/bin/env python3
"""
Text-only inference with the repo's VideoLLaMA3 stack (no images/video).
Loads the model the same way as evaluation/evaluate.py (INFERENCES + model_init),
builds inputs like videollama3/infer.py (conversation -> processor), and writes
instruction, input, output, and generation settings to a JSON file.

Unless --instruction is set, the default output is deterministic: a JSON array
(written as a native list in the output file) of temporal windows (start, end,
duration, window_type, action_indices, action_labels, metadata including
relevance_rank and query_lexical_score), ordered by relevance to the query.
No segment-selection or confidence LLM calls run in that mode.

With --instruction, the first conversation uses your custom instruction; optional
segment and confidence passes follow --skip-segment-proposal / --skip-confidence-refine.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# This file lives in evaluation/; repo root is one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CLASSES_FILE = _REPO_ROOT / "data" / "TSU_Action_list.txt"
_DEFAULT_OTB_JSON = _REPO_ROOT / "data" / "smarthome.json"
_DEFAULT_QA_GROUND_TRUTH_FILE = _REPO_ROOT / "data" / "test_balanced.txt"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Lines look like: c012 Tidying up a table
_CLASS_LINE_RE = re.compile(r"^(c\d{3})\s+(.+)$")
_CODE_RE = re.compile(r"\bc\d{3}\b")

# --- Lightweight NLP without extra deps: stopwords + verb lexicon + heuristics ---

_STOPWORDS = frozenset(
    """
    a an the and or but if as at by for from in into of on onto off out over to with
    without about above after again against all am any are around be been being both
    can could did do does doing done down each few further had has have having he her
    here hers herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now o of off on once only or other our ours ourselves out over
    own same she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself
    someone something somewhere sometime sometimes somehow somewhat
    near nearby behind beside front next between among along during before after
    within below above across against beyond despite except inside outside until
    """.split()
)

# Common English verbs / action lemmas; gerunds included; unknown -ing/-ed handled in extract_verb_terms.
_VERB_LEXICON = frozenset(
    """
    be have do say get make go know take see come think look want give use find tell ask
    work seem feel try leave call put mean keep let begin show hear play run move live
    stand sit meet bring write provide win lose pay open close hold walk talk laugh smile
    watch read eat drink pour wash throw tidy fix grasp reach grab turn snuggle dress
    undress cook sneeze awaken lie talking laughing smiling watching reading holding
    putting taking opening closing throwing tidying fixing working playing lying sitting
    standing walking running sneezing awakening eating drinking pouring washing
    """.split()
)

# Skip when treating as object nouns (too generic for substring matching).
_GENERIC_OBJECT_SKIP = frozenset(
    "person people someone somebody something somewhere thing things anyone anything".split()
)


def _tokenize_query(q: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-z]+)?", q.lower())


def _morph_variants(w: str) -> set[str]:
    """Variants for substring matching against gerunds and plurals in class names."""
    w = w.lower().strip()
    out: set[str] = set()
    if len(w) < 2:
        return out
    out.add(w)
    if len(w) > 4 and w.endswith("ing"):
        stem = w[:-3]
        if len(stem) >= 2:
            out.add(stem)
    if len(w) > 3 and w.endswith("ed"):
        out.add(w[:-2])
        if len(w) > 4 and w[-3] == w[-4]:  # stopped -> stop
            out.add(w[:-3])
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        out.add(w[:-1])
    return {t for t in out if len(t) >= 2}


def extract_noun_terms(query: str) -> list[str]:
    """Heuristic object nouns: content tokens not classified as verbs."""
    tokens = _tokenize_query(query)
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tokens:
        if t in _STOPWORDS or t in _GENERIC_OBJECT_SKIP:
            continue
        if len(t) < 2:
            continue
        is_verb = t in _VERB_LEXICON
        if not is_verb and len(t) > 4 and (t.endswith("ing") or t.endswith("ed")):
            is_verb = True
        if is_verb:
            continue
        if len(t) < 3 and t not in {"tv", "pc"}:
            continue
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def extract_verb_terms(query: str) -> list[str]:
    """Heuristic verbs: lexicon hits and clear -ing/-ed action forms."""
    tokens = _tokenize_query(query)
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tokens:
        if t in _STOPWORDS:
            continue
        if len(t) < 2:
            continue
        is_verb = t in _VERB_LEXICON
        if not is_verb and len(t) > 4 and (t.endswith("ing") or t.endswith("ed")):
            is_verb = True
        if not is_verb:
            continue
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def subset_classes_by_substrings(
    classes: list[tuple[str, str, str]],
    terms: list[str],
) -> list[str]:
    """
    Return class codes whose names contain any morphological variant of any term
    (case-insensitive substring).
    """
    if not terms:
        return []
    match_strings: set[str] = set()
    for term in terms:
        match_strings |= _morph_variants(term)
    codes: list[str] = []
    seen: set[str] = set()
    for code, _num, name in classes:
        lower = name.lower()
        if any(m and m in lower for m in match_strings):
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def load_otb_classes(path: Path) -> list[tuple[str, str, str]]:
    """
    Parse OTB action-list file. Returns list of (class_code, numeric_id, name).
    numeric_id is the decimal string without leading zeros (e.g. c012 -> "12", c000 -> "0").
    """
    text = path.read_text(encoding="utf-8")
    out: list[tuple[str, str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _CLASS_LINE_RE.match(line)
        if not m:
            continue
        code, name = m.group(1), m.group(2).strip()
        num = str(int(code[1:], 10))
        out.append((code, num, name))
    return out


def load_otb_annotations(path: Path) -> dict:
    """Load OTB annotation JSON as a dict keyed by video_id."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data).__name__}")
    return data


def get_video_actions_with_timestamps(
    video_id: str,
    annotations: dict,
    classes: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, float, float]], float | None]:
    """
    Build per-video action timeline from OTB annotation JSON action entries:
    [class_idx, start_sec, end_sec] -> (class_code, class_name, start_sec, end_sec).
    """
    rec = annotations.get(video_id, {})
    actions = rec.get("actions", []) if isinstance(rec, dict) else []
    duration = rec.get("duration") if isinstance(rec, dict) else None

    idx_to_class = {int(num): (code, name) for code, num, name in classes}
    out: list[tuple[str, str, float, float]] = []
    seen: set[tuple[str, float, float]] = set()
    for ann in actions:
        if not isinstance(ann, list) or len(ann) < 3:
            continue
        class_idx, start_sec, end_sec = ann[0], ann[1], ann[2]
        if not isinstance(class_idx, int):
            continue
        if class_idx not in idx_to_class:
            continue
        try:
            start_val = float(start_sec)
            end_val = float(end_sec)
        except (TypeError, ValueError):
            continue
        if end_val < start_val:
            continue
        code, name = idx_to_class[class_idx]
        key = (code, start_val, end_val)
        if key in seen:
            continue
        seen.add(key)
        out.append((code, name, start_val, end_val))

    out.sort(key=lambda x: (x[2], x[3], x[0]))
    return out, (float(duration) if isinstance(duration, (int, float)) else None)


def build_code_to_timeline_segments(
    video_actions_timeline: list[tuple[str, str, float, float]],
) -> dict[str, list[dict[str, float]]]:
    """
    Map each class code to all [start_sec, end_sec] occurrences in the video (ordered by time).
    """
    code_to_segments: dict[str, list[dict[str, float]]] = {}
    for code, _name, start_sec, end_sec in video_actions_timeline:
        seg = {"start_sec": float(start_sec), "end_sec": float(end_sec)}
        if code not in code_to_segments:
            code_to_segments[code] = []
        code_to_segments[code].append(seg)
    return code_to_segments


@dataclass
class ActionInterval:
    label: str
    start: float
    end: float


@dataclass
class TemporalWindow:
    start: float
    end: float
    action_indices: list[int]
    action_labels: list[str]
    window_type: str
    duration: float
    metadata: dict


def normalize_actions(actions: list[dict]) -> list[ActionInterval]:
    """Convert raw dicts to ActionInterval; skip invalid rows; sort by start then end."""
    out: list[ActionInterval] = []
    for d in actions:
        if not isinstance(d, dict):
            continue
        label = d.get("label")
        if label is None:
            label = d.get("name")
        if not isinstance(label, str) or not label.strip():
            continue
        s_raw = d.get("start")
        if s_raw is None:
            s_raw = d.get("start_sec")
        e_raw = d.get("end")
        if e_raw is None:
            e_raw = d.get("end_sec")
        if s_raw is None or e_raw is None:
            continue
        try:
            s = float(s_raw)
            e = float(e_raw)
        except (TypeError, ValueError):
            continue
        if s > e:
            continue
        out.append(ActionInterval(label=label.strip(), start=s, end=e))
    out.sort(key=lambda x: (x.start, x.end))
    return out


def build_atomic_windows(actions: list[ActionInterval]) -> list[TemporalWindow]:
    """One temporal window per normalized action (window_type \"atomic\")."""
    windows: list[TemporalWindow] = []
    for i, a in enumerate(actions):
        dur = a.end - a.start
        labels = [a.label]
        windows.append(
            TemporalWindow(
                start=a.start,
                end=a.end,
                action_indices=[i],
                action_labels=labels,
                window_type="atomic",
                duration=dur,
                metadata={
                    "num_actions": 1,
                    "source_query": "",
                    "start_action_label": labels[0],
                    "end_action_label": labels[-1],
                },
            )
        )
    return windows


def build_merged_windows(
    actions: list[ActionInterval],
    max_merge: int = 3,
    max_gap: float = 5.0,
) -> list[TemporalWindow]:
    """Merged windows over 2..max_merge consecutive actions if adjacent gaps <= max_gap."""
    n = len(actions)
    out: list[TemporalWindow] = []
    if max_merge < 2 or n < 2:
        return out
    for i in range(n):
        for L in range(2, min(max_merge, n - i) + 1):
            ok = True
            for k in range(i, i + L - 1):
                if actions[k + 1].start - actions[k].end > max_gap:
                    ok = False
                    break
            if not ok:
                break
            labels = [actions[j].label for j in range(i, i + L)]
            indices = list(range(i, i + L))
            start = actions[i].start
            end = actions[i + L - 1].end
            dur = end - start
            out.append(
                TemporalWindow(
                    start=start,
                    end=end,
                    action_indices=indices,
                    action_labels=labels,
                    window_type="merged",
                    duration=dur,
                    metadata={
                        "num_actions": L,
                        "source_query": "",
                        "start_action_label": labels[0],
                        "end_action_label": labels[-1],
                    },
                )
            )
    return out


def build_context_windows(
    actions: list[ActionInterval],
    video_length: float,
    context_before: float = 3.0,
    context_after: float = 3.0,
) -> list[TemporalWindow]:
    """Per-action windows expanded by context; clipped to [0, video_length]."""
    vl = float(video_length)
    out: list[TemporalWindow] = []
    for i, a in enumerate(actions):
        s = max(0.0, a.start - float(context_before))
        e = min(vl, a.end + float(context_after))
        if e < s:
            s, e = e, s
        dur = e - s
        labels = [a.label]
        out.append(
            TemporalWindow(
                start=s,
                end=e,
                action_indices=[i],
                action_labels=labels,
                window_type="context",
                duration=dur,
                metadata={
                    "num_actions": 1,
                    "source_query": "",
                    "start_action_label": labels[0],
                    "end_action_label": labels[-1],
                },
            )
        )
    return out


def deduplicate_windows(windows: list[TemporalWindow], tol: float = 1e-6) -> list[TemporalWindow]:
    """Drop duplicates with same (start, end, action_indices, window_type) within tol on times."""
    seen: set[tuple] = set()
    out: list[TemporalWindow] = []
    for w in windows:
        if tol <= 0:
            rs, re = w.start, w.end
        else:
            rs = round(w.start / tol) * tol
            re = round(w.end / tol) * tol
        key = (rs, re, tuple(w.action_indices), w.window_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


def generate_candidate_windows(
    query: str,
    actions: list[dict],
    video_length: float,
    max_merge: int = 3,
    max_gap: float = 5.0,
    context_before: float = 3.0,
    context_after: float = 3.0,
) -> list[TemporalWindow]:
    normalized = normalize_actions(actions)
    if not normalized:
        return []
    vl = float(video_length)
    atomic = build_atomic_windows(normalized)
    merged = build_merged_windows(normalized, max_merge=max_merge, max_gap=max_gap)
    context = build_context_windows(
        normalized,
        vl,
        context_before=context_before,
        context_after=context_after,
    )
    combined = atomic + merged + context
    for w in combined:
        w.metadata["source_query"] = query
    return deduplicate_windows(combined)


def window_to_dict(window: TemporalWindow) -> dict:
    """JSON-serializable dict for a TemporalWindow."""
    return {
        "start": window.start,
        "end": window.end,
        "action_indices": list(window.action_indices),
        "action_labels": list(window.action_labels),
        "window_type": window.window_type,
        "duration": window.duration,
        "metadata": dict(window.metadata),
    }


def order_temporal_windows_by_query_relevance(
    question: str,
    windows: list[TemporalWindow],
) -> list[TemporalWindow]:
    """Sort windows by descending query relevance (_temporal_window_rank_key)."""
    return sorted(
        windows,
        key=lambda w: _temporal_window_rank_key(w, question),
        reverse=True,
    )


def compute_ordered_temporal_windows_for_query(
    query: str,
    video_actions_timeline: list[tuple[str, str, float, float]],
    video_duration: float | None,
) -> list[dict]:
    """
    All candidate temporal windows (atomic, merged, context), deduplicated, then sorted
    by relevance to the query. Each dict includes metadata.relevance_rank (1 = best) and
    metadata.query_lexical_score.
    """
    if not video_actions_timeline:
        return []
    raw_actions = [
        {"label": name, "start_sec": float(s), "end_sec": float(e)}
        for _code, name, s, e in video_actions_timeline
    ]
    vl = (
        float(video_duration)
        if video_duration is not None
        else max(float(e) for _c, _n, _s, e in video_actions_timeline)
    )
    wins = generate_candidate_windows(query, raw_actions, vl)
    if not wins:
        return []
    ordered = order_temporal_windows_by_query_relevance(query, wins)
    for i, w in enumerate(ordered, start=1):
        w.metadata["relevance_rank"] = i
        w.metadata["query_lexical_score"] = _score_temporal_window_for_query(w, query)
    return [window_to_dict(w) for w in ordered]


def action_dicts_with_timestamps(
    codes: list[str],
    code_to_name: dict[str, str],
    code_to_segments: dict[str, list[dict[str, float]]] | None,
) -> list[dict]:
    """One entry per code in order; includes segments when ground-truth timeline is available."""
    out: list[dict] = []
    for c in codes:
        d: dict = {"code": c, "name": code_to_name[c]}
        segs = (code_to_segments or {}).get(c) or []
        if segs:
            d["segments"] = segs
            if len(segs) == 1:
                d["start_sec"] = segs[0]["start_sec"]
                d["end_sec"] = segs[0]["end_sec"]
        out.append(d)
    return out


def _merge_intervals_sorted(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping or touching intervals (closed [s,e])."""
    iv = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    if not iv:
        return []
    out: list[tuple[float, float]] = [iv[0]]
    for s, e in iv[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def _subtract_intervals(
    s: float, e: float, blocks: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return maximal sub-intervals of [s, e] not covered by the union of blocks (blocks merged)."""
    if e <= s:
        return []
    merged = _merge_intervals_sorted(blocks)
    cur: list[tuple[float, float]] = [(float(s), float(e))]
    for bs, be in merged:
        if be <= s or bs >= e:
            continue
        lo, hi = max(bs, s), min(be, e)
        if hi <= lo:
            continue
        nxt: list[tuple[float, float]] = []
        for cs, ce in cur:
            if hi <= cs or lo >= ce:
                nxt.append((cs, ce))
                continue
            if cs < lo:
                nxt.append((cs, min(lo, ce)))
            if ce > hi:
                nxt.append((max(hi, cs), ce))
        cur = _merge_intervals_sorted(nxt)
    return cur


def non_overlapping_lookup_windows_for_subset(
    codes: list[str],
    code_to_segments: dict[str, list[dict[str, float]]] | None,
) -> list[dict[str, float | str]]:
    """
    For actions in this subset (in list order), assign exclusive [start_sec, end_sec] pieces so
    no two emitted windows overlap: later actions lose overlap to earlier ones (greedy).
    Each window is one contiguous interval for one action_code.
    """
    if not codes:
        return []
    ct = code_to_segments or {}
    assigned: list[tuple[float, float]] = []
    out: list[dict[str, float | str]] = []
    for code in codes:
        segs = ct.get(code) or []
        for seg in segs:
            s = float(seg["start_sec"])
            e = float(seg["end_sec"])
            merged = _merge_intervals_sorted(assigned)
            for rs, re in _subtract_intervals(s, e, merged):
                if re - rs <= 1e-9:
                    continue
                out.append(
                    {
                        "action_code": code,
                        "start_sec": rs,
                        "end_sec": re,
                    }
                )
                assigned.append((rs, re))
    return out


def build_ground_truth_action_record(
    gt_code: str | None,
    code_to_name: dict[str, str],
    code_to_segments: dict[str, list[dict[str, float]]] | None,
) -> dict | None:
    """Resolve optional GT class code to name + timeline segments from detections."""
    if not gt_code or not gt_code.strip():
        return None
    code = gt_code.strip()
    if code not in code_to_name:
        return {
            "code": code,
            "name": None,
            "segments": [],
            "error": f"code {code!r} not in class list / video action set",
        }
    segs = (code_to_segments or {}).get(code) or []
    rec: dict = {
        "code": code,
        "name": code_to_name[code],
        "segments": segs,
    }
    if len(segs) == 1:
        rec["start_sec"] = segs[0]["start_sec"]
        rec["end_sec"] = segs[0]["end_sec"]
    return rec


def normalize_qa_question(q: str) -> str:
    """Match OTB questions regardless of extra whitespace or final punctuation."""
    t = q.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.rstrip("?.!")
    return t.lower()


def load_qa_ground_truth_file(path: Path) -> dict:
    """Load OTB-style JSON: top-level keys are question ids; values have question, answer, video_id."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}, got {type(data).__name__}")
    return data


def lookup_qa_ground_truth(
    video_id: str,
    question: str,
    qa_data: dict,
) -> tuple[dict | None, str | None]:
    """Find record where video_id and normalized question match. Returns (record, question_id key)."""
    if not video_id or not qa_data:
        return None, None
    nq = normalize_qa_question(question)
    for qid, rec in qa_data.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("video_id") != video_id:
            continue
        rq = rec.get("question")
        if not isinstance(rq, str):
            continue
        if normalize_qa_question(rq) == nq:
            return rec, str(qid)
    return None, None


def build_ground_truth_output_bundle(
    *,
    video_id: str,
    question: str,
    qa_path: Path | None,
    qa_data: dict | None,
    gt_action_code_cli: str | None,
    code_to_name: dict[str, str],
    code_to_segments: dict[str, list[dict[str, float]]] | None,
) -> tuple[str | None, dict | None]:
    """
    Resolve string ground_truth_answer from OTB-style JSON (video_id + question match),
    optional OTB action from --gt-action-code, and return (answer_str, composite_dict).
    """
    qa_rec: dict | None = None
    question_id: str | None = None
    if qa_data and video_id.strip():
        qa_rec, question_id = lookup_qa_ground_truth(video_id.strip(), question, qa_data)

    gt_answer: str | None = None
    if qa_rec is not None:
        ans = qa_rec.get("answer")
        if isinstance(ans, str) and ans.strip():
            gt_answer = ans.strip()

    otb_action: dict | None = None
    if gt_action_code_cli and gt_action_code_cli.strip():
        otb_action = build_ground_truth_action_record(
            gt_action_code_cli.strip(),
            code_to_name,
            code_to_segments,
        )

    composite: dict = {}
    if gt_answer is not None:
        composite["ground_truth_answer"] = gt_answer
    if qa_rec is not None:
        if question_id:
            composite["question_id"] = question_id
        if qa_path is not None:
            composite["qa_source"] = str(qa_path.resolve())
    if otb_action is not None:
        composite["otb_action"] = otb_action

    if not composite:
        return None, None
    return gt_answer, composite


_WORKING_AT_A_RE = re.compile(r"^Working at a (.+)$", re.IGNORECASE)


def build_first_pass_system_instruction(*, has_action_timeline: bool) -> str:
    """Strict grounding rules so answers come from OTB action names, not hallucination."""
    base = [
        "You answer questions about a video using ONLY the detected action list you will receive.",
        "Rules:",
        "- Do not invent actions, objects, or times that are not supported by the action names and intervals.",
        '- For "what was the person working on" (or similar), the answer must be a single short noun or noun phrase taken from the relevant action NAME text — e.g. if the list includes "Working at a table", a correct minimal answer is: table.',
        "- Prefer one word when the action name clearly names one object or surface (table, dish, towel, cup, …).",
        "- If the question refers to order in time (before/after/between), use intervals [start_sec, end_sec] to decide which action names apply; still name only what those action strings support.",
        "- If the list cannot answer the question, reply exactly: insufficient evidence",
        "- After the word Answer: output a single line: the answer only, no explanation.",
    ]
    if not has_action_timeline:
        base.append("- No action list was provided; reply: insufficient evidence")
    return "\n".join(base)


def suggest_short_answer_from_actions(
    question: str,
    video_actions_timeline: list[tuple[str, str, float, float]],
) -> dict | None:
    """
    Deterministic short answer from action class names (for eval / when the model drifts).
    Handles common OTB-style "what were they working on (after …)?" by parsing "Working at a …".
    """
    if not video_actions_timeline:
        return None
    q = question.strip().lower()
    wash_end: float | None = None
    if "after" in q:
        se_towel: float | None = None
        se_any: float | None = None
        for _c, name, _s, e in video_actions_timeline:
            nl = name.lower()
            if "washing" in nl and "towel" in nl:
                se_towel = max(se_towel or 0.0, float(e))
            if "wash" in nl or "washing" in nl:
                se_any = max(se_any or 0.0, float(e))
        wash_end = se_towel if se_towel is not None else se_any

    t_ref = wash_end if wash_end is not None else 0.0
    ordered = sorted(video_actions_timeline, key=lambda x: (x[2], x[3], x[0]))

    for code, name, s, e in ordered:
        if "after" in q and wash_end is not None and e <= wash_end:
            continue
        m = _WORKING_AT_A_RE.match(name.strip())
        if not m:
            continue
        noun = m.group(1).strip().rstrip(".")
        if not noun:
            continue
        return {
            "short_answer": noun,
            "source_action_code": code,
            "source_action_name": name,
            "interval_start_sec": s,
            "interval_end_sec": e,
            "reference_after_sec": t_ref if wash_end is not None else None,
        }
    return None


def build_first_pass_answer_user_message(
    question: str,
    video_id: str | None,
    video_duration: float | None,
    video_actions_timeline: list[tuple[str, str, float, float]],
) -> str:
    """User message for a direct answer given the question + detected action timeline."""
    lines = [
        "Use the detected actions below as the only evidence (you do not see the video).",
        "",
    ]
    if video_id:
        lines.append(f"Video ID: {video_id}")
    if video_duration is not None:
        lines.append(f"Video duration (seconds): {video_duration:.2f}")
    lines.append("")
    lines.append("Detected actions (class code — name [start_sec, end_sec]):")
    if video_actions_timeline:
        for code, name, start_sec, end_sec in video_actions_timeline:
            lines.append(f"  {code} — {name} [{start_sec:.2f}, {end_sec:.2f}]")
    else:
        lines.append("  (none)")
    lines.extend(
        [
            "",
            f"Question: {question}",
            "",
            "Answer (one line, minimal noun when possible; must follow the system rules):",
        ]
    )
    return "\n".join(lines)


def build_temporal_windows_output_instruction(*, has_action_timeline: bool) -> str:
    """Describes the sole default output: a JSON array of temporal windows ordered by query relevance."""
    lines = [
        "Output specification — this is the only output (no other keys, text, or prose):",
        "Return exactly one JSON array.",
        "Each element is one temporal window object with:",
        "- start: float (seconds)",
        "- end: float (seconds)",
        "- duration: float (end minus start)",
        '- window_type: \"atomic\" | \"merged\" | \"context\"',
        "- action_indices: list of int (zero-based indices into detected actions sorted by time)",
        "- action_labels: list of str",
        "- metadata: object including at least num_actions, source_query, start_action_label, end_action_label, relevance_rank (1 = most relevant to the query), query_lexical_score",
        "Order the array by descending relevance to the query (most relevant window first).",
        "Windows are derived only from detected action names and [start_sec, end_sec] intervals (atomic, optional short merged chains, and local context expansions).",
    ]
    if not has_action_timeline:
        lines.append("If there are no detected actions, return [].")
    return "\n".join(lines)


def build_segment_selection_system_instruction(*, has_action_timeline: bool) -> str:
    """Same as build_temporal_windows_output_instruction (kept for imports in other evaluation scripts)."""
    return build_temporal_windows_output_instruction(has_action_timeline=has_action_timeline)


def build_temporal_windows_output_user_message(
    question: str,
    video_id: str | None,
    video_duration: float | None,
    video_actions_timeline: list[tuple[str, str, float, float]],
) -> str:
    """User-side context for default mode; the pipeline fills the output deterministically."""
    lines = [
        "Temporal window list is computed from the query and the detected action timeline below.",
        "",
    ]
    if video_id:
        lines.append(f"Video ID: {video_id}")
    if video_duration is not None:
        lines.append(f"Video duration (seconds): {video_duration:.2f}")
    lines.append("")
    lines.append("Detected actions (class code — name [start_sec, end_sec]):")
    if video_actions_timeline:
        for code, name, start_sec, end_sec in video_actions_timeline:
            lines.append(f"  {code} — {name} [{start_sec:.2f}, {end_sec:.2f}]")
    else:
        lines.append("  (none)")
    lines.extend(["", f"Query: {question}", ""])
    return "\n".join(lines)


def build_segment_selection_user_message(
    question: str,
    video_id: str | None,
    video_duration: float | None,
    video_actions_timeline: list[tuple[str, str, float, float]],
    *,
    temporal_candidate_windows: list[dict] | None = None,
) -> str:
    """User payload when --instruction is set and segment LLM runs; includes optional ranked windows as hints."""
    lines = [
        "Context for temporal reasoning (custom instruction applies).",
        "",
    ]
    if video_id:
        lines.append(f"Video ID: {video_id}")
    if video_duration is not None:
        lines.append(f"Video duration (seconds): {video_duration:.2f}")
    lines.append("")
    lines.append("Detected actions (class code — name [start_sec, end_sec]):")
    if video_actions_timeline:
        for code, name, start_sec, end_sec in video_actions_timeline:
            lines.append(f"  {code} — {name} [{start_sec:.2f}, {end_sec:.2f}]")
    else:
        lines.append("  (none)")
    if temporal_candidate_windows:
        lines.append("")
        lines.append(
            "Query-relevance-ordered temporal windows (reference; same schema as default output):"
        )
        for i, w in enumerate(temporal_candidate_windows):
            lines.append(f"  [{i}] {json.dumps(w, ensure_ascii=False)}")
    lines.extend(["", f"Query: {question}", ""])
    return "\n".join(lines)


def _segment_interval(
    seg: dict | None,
) -> tuple[float | None, float | None]:
    if not seg:
        return None, None
    s, e = seg.get("start_time"), seg.get("end_time")
    if s is None or e is None:
        return None, None
    try:
        return float(s), float(e)
    except (TypeError, ValueError):
        return None, None


def segments_are_distinct(
    a: dict | None,
    b: dict | None,
    *,
    tol_sec: float = 0.05,
) -> bool:
    """Return True if intervals differ by more than tol_sec on either endpoint."""
    sa, ea = _segment_interval(a)
    sb, eb = _segment_interval(b)
    if sa is None or ea is None or sb is None or eb is None:
        return True
    return abs(sa - sb) > tol_sec or abs(ea - eb) > tol_sec


def nudge_segment_interval(
    seg: dict | None,
    *,
    video_duration: float | None,
) -> dict | None:
    """Widen/shift [start,end] slightly so the interval differs from a duplicate proposal."""
    if not seg:
        return seg
    s, e = _segment_interval(seg)
    if s is None or e is None:
        return seg
    pad = 0.25
    ns = max(0.0, s - pad)
    ne = e + pad
    if video_duration is not None:
        ne = min(ne, float(video_duration))
    if ne <= ns:
        ne = ns + 0.1
    out = dict(seg)
    out["start_time"] = ns
    out["end_time"] = ne
    j = str(out.get("justification") or "").strip()
    out["justification"] = (j + " [interval widened to differ from prior proposal]").strip()
    return out


def collect_actions_overlapping_segment(
    video_actions_timeline: list[tuple[str, str, float, float]],
    start_sec: float,
    end_sec: float,
) -> list[tuple[str, str, float, float]]:
    out: list[tuple[str, str, float, float]] = []
    for row in video_actions_timeline:
        _c, _n, s, e = row
        if e < start_sec or s > end_sec:
            continue
        out.append(row)
    return out


def build_textual_description_for_segment(
    segment: dict,
    overlapping_actions: list[tuple[str, str, float, float]],
) -> str:
    """Evidence text for confidence checking and for the final answer pass."""
    parts: list[str] = []
    j = str(segment.get("justification") or "").strip()
    if j:
        parts.append(f"Segment rationale: {j}")
    s, e = _segment_interval(segment)
    if s is not None and e is not None:
        parts.append(f"Time window (seconds): [{s:.2f}, {e:.2f}]")
    if overlapping_actions:
        parts.append("Detected actions overlapping this window (code — name [start, end]):")
        for code, name, a_s, a_e in overlapping_actions:
            parts.append(f"  {code} — {name} [{a_s:.2f}, {a_e:.2f}]")
    else:
        parts.append("No detected actions overlap this time window.")
    return "\n".join(parts).strip()


def build_confidence_system_instruction() -> str:
    return "\n".join(
        [
            "You judge whether the given evidence description is enough to answer the user query.",
            "The description is text-only (no video); base your judgment only on whether the listed actions and time window plausibly support a specific answer to the query.",
            "Rules:",
            "- Output exactly one JSON object and nothing else.",
            '- Format: {"confidence": <0 or 1>, "reason": "<one short sentence>"}',
            "- confidence must be integer 0 or 1 only.",
            "- Use confidence 1 if the description is sufficient to answer the query (clear enough evidence).",
            "- Use confidence 0 if a different time window or additional evidence would likely be needed.",
        ]
    )


def build_confidence_user_message(
    question: str,
    evidence_description: str,
    video_id: str | None,
    video_duration: float | None,
) -> str:
    lines = [
        "Decide if the following evidence description is sufficient to answer the query.",
        "",
    ]
    if video_id:
        lines.append(f"Video ID: {video_id}")
    if video_duration is not None:
        lines.append(f"Video duration (seconds): {video_duration:.2f}")
    lines.extend(
        [
            "",
            f"Query: {question}",
            "",
            "Evidence description:",
            evidence_description,
            "",
            "Return only the JSON object.",
        ]
    )
    return "\n".join(lines)


def parse_confidence_json_from_model_output(text: str) -> tuple[int | None, str]:
    """Parse {'confidence': 0|1, 'reason': ...}. Returns (confidence or None, reason string)."""
    if not text or not text.strip():
        return None, ""
    raw = text.strip()
    candidates: list[str] = [raw]
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
        conf = obj.get("confidence")
        reason_raw = obj.get("reason", "")
        reason = str(reason_raw).strip() if reason_raw is not None else ""
        try:
            ci = int(conf)
        except (TypeError, ValueError):
            continue
        if ci in (0, 1):
            return ci, reason
    return None, ""


def build_segment_selection_system_instruction_retry(*, has_action_timeline: bool) -> str:
    base = build_segment_selection_system_instruction(has_action_timeline=has_action_timeline)
    extra = [
        "",
        "Additional constraints for this request:",
        "- A previous answer was judged insufficient.",
        "- If your instruction expects a JSON array of windows, output a NEW array that differs (different ordering or window boundaries).",
        "- If your instruction expects a single segment object, propose a NEW interval (change start/end meaningfully).",
    ]
    return base + "\n" + "\n".join(extra)


def build_segment_selection_user_message_retry(
    question: str,
    video_id: str | None,
    video_duration: float | None,
    video_actions_timeline: list[tuple[str, str, float, float]],
    previous_segment: dict,
    previous_description: str,
    *,
    temporal_candidate_windows: list[dict] | None = None,
) -> str:
    """Ask the segment LLM for a different window given prior clip + description."""
    ps, pe = _segment_interval(previous_segment)
    prev_seg_line = (
        f"[{ps:.2f}, {pe:.2f}]"
        if ps is not None and pe is not None
        else "null"
    )
    lines = [
        "The previously selected segment and its text evidence were not enough to answer the query.",
        "Propose a DIFFERENT single continuous time segment that is more likely to contain the answer.",
        "",
    ]
    if video_id:
        lines.append(f"Video ID: {video_id}")
    if video_duration is not None:
        lines.append(f"Video duration (seconds): {video_duration:.2f}")
    lines.extend(
        [
            "",
            "Previous segment that was clipped (seconds): " + prev_seg_line,
            "",
            "Previous segment evidence description:",
            previous_description,
            "",
            "Detected actions (class code — name [start_sec, end_sec]):",
        ]
    )
    if video_actions_timeline:
        for code, name, start_sec, end_sec in video_actions_timeline:
            lines.append(f"  {code} — {name} [{start_sec:.2f}, {end_sec:.2f}]")
    else:
        lines.append("  (none)")
    if temporal_candidate_windows:
        lines.append("")
        lines.append(
            "Temporal proposal candidates (JSON objects; start/end in seconds, aligned to sorted detected actions):"
        )
        for i, w in enumerate(temporal_candidate_windows):
            lines.append(f"  [{i}] {json.dumps(w, ensure_ascii=False)}")
    lines.extend(
        [
            "",
            f"Query: {question}",
            "",
            "Return only the JSON object for the NEW segment (must differ from the previous interval).",
        ]
    )
    return "\n".join(lines)


def build_first_pass_answer_user_message_with_segment_history(
    question: str,
    video_id: str | None,
    video_duration: float | None,
    video_actions_timeline: list[tuple[str, str, float, float]],
    *,
    previous_segment_descriptions: list[str],
    current_segment_description: str | None,
) -> str:
    """Like build_first_pass_answer_user_message but adds prior + current segment evidence."""
    base = build_first_pass_answer_user_message(
        question=question,
        video_id=video_id,
        video_duration=video_duration,
        video_actions_timeline=video_actions_timeline,
    )
    extra_blocks: list[str] = []
    if previous_segment_descriptions:
        extra_blocks.append(
            "Previous segment evidence descriptions (insufficient alone; use together with the current segment):"
        )
        for i, desc in enumerate(previous_segment_descriptions, start=1):
            extra_blocks.append(f"--- Previous segment {i} ---\n{desc}")
    if current_segment_description and str(current_segment_description).strip():
        extra_blocks.append(
            "Current segment evidence description (primary):\n" + current_segment_description.strip()
        )
    if not extra_blocks:
        return base
    insertion = "\n\n".join(extra_blocks)
    split_key = "\n\nAnswer (one line"
    if split_key in base:
        a, b = base.split(split_key, 1)
        return a + "\n\n" + insertion + split_key + b
    return base + "\n\n" + insertion


def parse_segment_json_from_model_output(
    text: str,
    *,
    video_duration: float | None,
) -> dict | None:
    """
    Parse model output into {'start_time': float|None, 'end_time': float|None, 'justification': str}.
    Accepts either start_time/end_time or start_sec/end_sec from the model.
    """
    if not text or not text.strip():
        return None

    raw = text.strip()
    candidates: list[str] = [raw]
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        candidates.insert(0, m.group(0))

    payload: dict | None = None
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            payload = obj
            break
    if payload is None:
        return None

    s_raw = payload.get("start_time", payload.get("start_sec"))
    e_raw = payload.get("end_time", payload.get("end_sec"))
    justification_raw = payload.get("justification")
    justification = str(justification_raw).strip() if justification_raw is not None else ""
    if s_raw is None and e_raw is None:
        return {
            "start_time": None,
            "end_time": None,
            "justification": justification or "No segment selected.",
        }
    if s_raw is None or e_raw is None:
        return None
    try:
        s = float(s_raw)
        e = float(e_raw)
    except (TypeError, ValueError):
        return None

    if video_duration is not None:
        s = max(0.0, min(s, float(video_duration)))
        e = max(0.0, min(e, float(video_duration)))
    else:
        s = max(0.0, s)
        e = max(0.0, e)

    if e < s:
        s, e = e, s
    return {
        "start_time": s,
        "end_time": e,
        "justification": justification or "Selected based on detected action timeline.",
    }


def _score_action_name_match(action_name: str, query_terms: list[str]) -> int:
    """Simple lexical score between query terms and one action name."""
    if not query_terms:
        return 0
    lower_name = action_name.lower()
    score = 0
    for t in query_terms:
        variants = _morph_variants(t)
        if any(v in lower_name for v in variants):
            score += 1
    return score


def _infer_temporal_relation(question: str) -> str:
    """Infer whether query targets before/during/after an action."""
    q = question.lower()
    if " before " in f" {q} ":
        return "before"
    if " after " in f" {q} ":
        return "after"
    if any(w in q for w in ("during", "while", "when")):
        return "during"
    return "during"


def _pick_target_action_instance(
    question: str,
    video_actions_timeline: list[tuple[str, str, float, float]],
) -> tuple[str, str, float, float] | None:
    """Pick the action instance most likely referenced by the query."""
    if not video_actions_timeline:
        return None

    q = question.lower()
    terms: list[str] = []
    temporal_words = {"before", "after", "during", "while", "when", "then"}
    for t in extract_verb_terms(question) + extract_noun_terms(question):
        if t not in temporal_words:
            terms.append(t)

    scored: list[tuple[int, tuple[str, str, float, float]]] = []
    for action in video_actions_timeline:
        code, name, _s, _e = action
        score = _score_action_name_match(name, terms)
        if code in q:
            score += 2
        scored.append((score, action))

    scored.sort(key=lambda x: (x[0], x[1][2], x[1][3]), reverse=True)
    best_score, _best_action = scored[0]
    if best_score <= 0:
        # If no lexical match is found, default to earliest action to keep output grounded.
        return sorted(video_actions_timeline, key=lambda x: (x[2], x[3], x[0]))[0]

    if any(w in q for w in ("last", "latest", "final")):
        scored_best = [a for s, a in scored if s == best_score]
        return sorted(scored_best, key=lambda x: (x[2], x[3], x[0]))[-1]
    if any(w in q for w in ("first", "earliest", "initial")):
        scored_best = [a for s, a in scored if s == best_score]
        return sorted(scored_best, key=lambda x: (x[2], x[3], x[0]))[0]
    return scored[0][1]


def _query_terms_for_window_scoring(question: str) -> list[str]:
    temporal_words = {"before", "after", "during", "while", "when", "then"}
    terms: list[str] = []
    for t in extract_verb_terms(question) + extract_noun_terms(question):
        if t not in temporal_words:
            terms.append(t)
    return terms


def _score_temporal_window_for_query(window: TemporalWindow, question: str) -> int:
    terms = _query_terms_for_window_scoring(question)
    if not terms:
        return 0
    score = 0
    for lab in window.action_labels:
        score += _score_action_name_match(lab, terms)
    return score


def _temporal_window_rank_key(window: TemporalWindow, question: str) -> tuple:
    """Higher is better: lexical score, then prefer atomic over merged over context, then tighter interval."""
    type_order = {"atomic": 0, "merged": 1, "context": 2}
    score = _score_temporal_window_for_query(window, question)
    inv_dur = 1.0 / (window.duration + 1e-9)
    return (score, -type_order.get(window.window_type, 9), inv_dur)


def _select_temporal_segment_from_actions_legacy(
    question: str,
    video_actions_timeline: list[tuple[str, str, float, float]],
    *,
    video_duration: float | None,
) -> dict | None:
    """
    Relation-based segment from action timeline (before/during/after target action).
    """
    if not video_actions_timeline:
        return {
            "start_time": None,
            "end_time": None,
            "justification": "No detected actions were provided.",
        }

    ordered = sorted(video_actions_timeline, key=lambda x: (x[2], x[3], x[0]))
    target = _pick_target_action_instance(question, ordered)
    if target is None:
        return None

    relation = _infer_temporal_relation(question)
    code, name, start_sec, end_sec = target

    if relation == "during":
        return {
            "start_time": float(start_sec),
            "end_time": float(end_sec),
            "justification": (
                f'Query refers to "{name}" ({code}); selected its detected interval '
                f"[{start_sec:.2f}, {end_sec:.2f}] for during-action evidence."
            ),
        }

    if relation == "before":
        previous = [a for a in ordered if a[3] <= start_sec]
        if previous:
            p_code, p_name, p_start, p_end = previous[-1]
            return {
                "start_time": float(p_start),
                "end_time": float(p_end),
                "justification": (
                    f'Query asks for before "{name}" ({code}); selected the closest prior '
                    f'action "{p_name}" ({p_code}) at [{p_start:.2f}, {p_end:.2f}].'
                ),
            }
        fallback_start = 0.0
        fallback_end = max(0.0, float(start_sec))
        return {
            "start_time": fallback_start,
            "end_time": fallback_end,
            "justification": (
                f'No detected action occurs before "{name}" ({code}); selected the pre-action '
                f"context window [{fallback_start:.2f}, {fallback_end:.2f}]."
            ),
        }

    # relation == "after"
    following = [a for a in ordered if a[2] >= end_sec]
    if following:
        n_code, n_name, n_start, n_end = following[0]
        return {
            "start_time": float(n_start),
            "end_time": float(n_end),
            "justification": (
                f'Query asks for after "{name}" ({code}); selected the closest subsequent '
                f'action "{n_name}" ({n_code}) at [{n_start:.2f}, {n_end:.2f}].'
            ),
        }

    fallback_start = float(end_sec)
    if video_duration is not None:
        fallback_end = max(fallback_start, float(video_duration))
    else:
        fallback_end = fallback_start
    return {
        "start_time": fallback_start,
        "end_time": fallback_end,
        "justification": (
            f'No detected action occurs after "{name}" ({code}); selected trailing context '
            f"from [{fallback_start:.2f}, {fallback_end:.2f}]."
        ),
    }


def select_temporal_segment_from_actions(
    question: str,
    video_actions_timeline: list[tuple[str, str, float, float]],
    *,
    video_duration: float | None,
) -> dict | None:
    """
    Build action-grounded temporal proposals (atomic / merged / context), score with the query,
    and pick a segment; fall back to relation-based legacy selection when no lexical match.
    """
    if not video_actions_timeline:
        return {
            "start_time": None,
            "end_time": None,
            "justification": "No detected actions were provided.",
        }

    raw_actions: list[dict] = [
        {"label": name, "start_sec": float(s), "end_sec": float(e)}
        for _code, name, s, e in video_actions_timeline
    ]
    vl = (
        float(video_duration)
        if video_duration is not None
        else max(float(e) for _c, _n, _s, e in video_actions_timeline)
    )
    candidates = generate_candidate_windows(question, raw_actions, vl)
    if not candidates:
        return {
            "start_time": None,
            "end_time": None,
            "justification": "No valid temporal proposals from detected actions.",
        }

    scored = [w for w in candidates if _score_temporal_window_for_query(w, question) > 0]
    if scored:
        best = max(scored, key=lambda w: _temporal_window_rank_key(w, question))
        labels_summary = ", ".join(best.action_labels)
        return {
            "start_time": float(best.start),
            "end_time": float(best.end),
            "justification": (
                f'Query aligned with candidate window type "{best.window_type}" over '
                f'[{labels_summary}] at [{best.start:.2f}, {best.end:.2f}] '
                f"(action-grounded proposals)."
            ),
        }

    return _select_temporal_segment_from_actions_legacy(
        question,
        video_actions_timeline,
        video_duration=video_duration,
    )


def build_semantic_subset_instruction(
    classes: list[tuple[str, str, str]],
    video_id: str | None = None,
    video_actions: list[tuple[str, str, float, float]] | None = None,
    video_duration: float | None = None,
    object_codes: list[str] | None = None,
    verb_codes: list[str] | None = None,
) -> str:
    """LLM prompt for Subset 3 only (semantic similarity to the query)."""
    lines = [
        "You must produce ONE subset of OTB classes for the user's question.",
        "",
        "Subset 3 (Semantic-based):",
        "- Using the meaning of the query (and paraphrases of its parts), list every class",
        "  from the provided list whose action is semantically similar or closely aligned",
        "  with what the query describes. Include all plausible matches; do not stop at a",
        "  single best class. Exclude classes that are clearly unrelated.",
        "",
        "Use only classes from the provided list. Remove duplicates. If there are no",
        "matches, output `none` under the section.",
        "",
        "Output exactly one section in this form:",
        "Subset 3: Semantic-based",
        "class_code — name",
        "...",
        "",
        "No JSON, explanations, or other sections. Only the header and class lines.",
    ]
    if video_actions:
        obj = object_codes or []
        vb = verb_codes or []
        lines.extend(
            [
                "",
                "Ground-truth timeline coverage (same video):",
                "Subset 1 (object-based) and Subset 2 (verb-based) are already fixed from the",
                "query text (codes may overlap between 1 and 2). After you answer, the pipeline",
                "places every timeline action whose class code is not in Subset 1, Subset 2,",
                "or Subset 3 into a separate bucket named Subset 4: Remaining actions.",
                "You do NOT output Subset 4; only output Subset 3 as specified above.",
                "",
                "Strive to put every semantically relevant timeline action into Subset 3 so that",
                "Subset 4 only holds timeline actions that are genuinely unrelated to the query.",
                "",
                f"Subset 1 (object-based) class codes already selected: {obj if obj else 'none'}",
                f"Subset 2 (verb-based) class codes already selected: {vb if vb else 'none'}",
                "",
                "Use this per-video action timeline as grounding context (actions happening in the video):",
                f"Video ID: {video_id or 'unknown'}",
            ]
        )
        if video_duration is not None:
            lines.append(f"Video duration (sec): {video_duration:.2f}")
        for code, name, start_sec, end_sec in video_actions:
            lines.append(f"  {code} — {name} [{start_sec:.2f}, {end_sec:.2f}]")
        lines.append(
            "Prioritize classes from this timeline whenever possible; still include any other provided class if semantically required."
        )
    else:
        for code, _num, name in classes:
            lines.append(f"  {code} — {name}")
    return "\n".join(lines)


def canonicalize_semantic_subset_output(
    output_text: str, classes: list[tuple[str, str, str]] | None
) -> tuple[str, list[str]]:
    """
    Parse LLM output for Subset 3 only; return formatted block and list of codes.
    """
    if not classes:
        return output_text, []

    code_to_name = {code: name for code, _num, name in classes}
    header = "Subset 3: Semantic-based"
    codes: list[str] = []
    seen: set[str] = set()
    current = False

    for raw in output_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("subset 3"):
            current = True
            continue
        if lower.startswith("subset ") and not lower.startswith("subset 3"):
            current = False
            continue
        if not current:
            continue
        for code in _CODE_RE.findall(line):
            if code in code_to_name and code not in seen:
                seen.add(code)
                codes.append(code)

    body_lines: list[str] = [header]
    if codes:
        for code in codes:
            body_lines.append(f"{code} — {code_to_name[code]}")
    else:
        body_lines.append("none")

    return "\n".join(body_lines), codes


def _format_class_line_with_optional_times(
    code: str,
    name: str,
    code_to_segments: dict[str, list[dict[str, float]]] | None,
) -> str:
    base = f"{code} — {name}"
    if not code_to_segments:
        return base
    segs = code_to_segments.get(code) or []
    if not segs:
        return base
    parts_t = []
    for s in segs:
        parts_t.append(f"[{s['start_sec']:.2f}, {s['end_sec']:.2f}]")
    return f"{base} {' '.join(parts_t)}"


def format_three_subsets(
    code_to_name: dict[str, str],
    object_codes: list[str],
    verb_codes: list[str],
    semantic_block: str,
    remaining_codes: list[str] | None = None,
    code_to_segments: dict[str, list[dict[str, float]]] | None = None,
) -> str:
    """Assemble final text; optional Subset 4 lists GT actions not in Subsets 1–3."""
    parts: list[str] = []

    parts.append("Subset 1: Object-based (nouns in action names)")
    if object_codes:
        for c in object_codes:
            parts.append(_format_class_line_with_optional_times(c, code_to_name[c], code_to_segments))
    else:
        parts.append("none")
    parts.append("")
    parts.append("Subset 2: Action/verb-based (verbs in action names)")
    if verb_codes:
        for c in verb_codes:
            parts.append(_format_class_line_with_optional_times(c, code_to_name[c], code_to_segments))
    else:
        parts.append("none")
    parts.append("")
    parts.extend(_format_semantic_block_with_optional_times(semantic_block, code_to_name, code_to_segments))

    if remaining_codes is not None:
        parts.append("")
        parts.append("Subset 4: Remaining actions (ground-truth timeline not in Subsets 1–3)")
        if remaining_codes:
            for c in remaining_codes:
                parts.append(_format_class_line_with_optional_times(c, code_to_name[c], code_to_segments))
        else:
            parts.append("none")

    return "\n".join(parts)


def _format_semantic_block_with_optional_times(
    semantic_block: str,
    code_to_name: dict[str, str],
    code_to_segments: dict[str, list[dict[str, float]]] | None,
) -> list[str]:
    """Rewrite semantic subset lines to append [start, end] when timeline is available."""
    if not code_to_segments:
        return semantic_block.splitlines()

    out_lines: list[str] = []
    in_subset3 = False
    for raw in semantic_block.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("subset 3"):
            in_subset3 = True
            out_lines.append(line)
            continue
        if lower.startswith("subset ") and not lower.startswith("subset 3"):
            in_subset3 = False
            out_lines.append(line)
            continue
        if not in_subset3 or stripped.lower() == "none" or not stripped:
            out_lines.append(line)
            continue
        codes_in_line = _CODE_RE.findall(stripped)
        if not codes_in_line:
            out_lines.append(line)
            continue
        code = codes_in_line[0]
        if code not in code_to_name:
            out_lines.append(line)
            continue
        out_lines.append(_format_class_line_with_optional_times(code, code_to_name[code], code_to_segments))
    return out_lines


def build_conversation(instruction: str | None, user_text: str) -> list[dict]:
    conv = []
    if instruction and instruction.strip():
        conv.append({"role": "system", "content": instruction.strip()})
    conv.append({"role": "user", "content": user_text.strip()})
    return conv



# --- Imported from v7 for primary pipelines ---

def _extract_content_terms(text: str) -> list[str]:
    """Content terms for matching action names (lightweight, no external NLP deps)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for t in _tokenize_query(text):
        if t in _STOPWORDS or t in _GENERIC_OBJECT_SKIP:
            continue
        if len(t) < 2:
            continue
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered

def _matched_terms_in_action_name(action_name: str, terms: list[str]) -> list[str]:
    """Return query terms whose morphological variants appear in the action name."""
    if not terms:
        return []
    lower = action_name.lower()
    hits: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if t in seen:
            continue
        variants = _morph_variants(t)
        if any(v in lower for v in variants):
            seen.add(t)
            hits.append(t)
    return hits

def _parse_temporal_query_hints(question: str) -> dict:
    """
    Parse lightweight temporal intent from query text.
    Supports anchors for: between X and Y, after X, before X, while/during X.
    """
    q = re.sub(r"\s+", " ", question.strip().lower())
    hints = {
        "mode": None,
        "anchor_terms_left": [],
        "anchor_terms_right": [],
        "prefer_first": any(k in q for k in (" first ", " earliest ", " beginning ", " initially ")),
        "prefer_last": any(k in q for k in (" last ", " final ", " end ", " finally ", " later ")),
    }
    padded = f" {q} "

    m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$|[.,;])", q)
    if m:
        hints["mode"] = "between"
        hints["anchor_terms_left"] = _extract_content_terms(m.group(1))
        hints["anchor_terms_right"] = _extract_content_terms(m.group(2))
        return hints

    m = re.search(r"\b(?:while|during)\s+(.+?)(?:\?|$|[.,;])", q)
    if m:
        hints["mode"] = "while"
        hints["anchor_terms_left"] = _extract_content_terms(m.group(1))
        return hints

    m = re.search(r"\bafter\s+(.+?)(?:\?|$|[.,;])", q)
    if m:
        hints["mode"] = "after"
        hints["anchor_terms_left"] = _extract_content_terms(m.group(1))
        return hints

    m = re.search(r"\bbefore\s+(.+?)(?:\?|$|[.,;])", q)
    if m:
        hints["mode"] = "before"
        hints["anchor_terms_left"] = _extract_content_terms(m.group(1))
        return hints

    # Common phrasing where punctuation is absent at the end.
    if hints["mode"] is None and " after " in padded:
        phrase = padded.split(" after ", 1)[1].strip()
        hints["mode"] = "after"
        hints["anchor_terms_left"] = _extract_content_terms(phrase)
    elif hints["mode"] is None and " before " in padded:
        phrase = padded.split(" before ", 1)[1].strip()
        hints["mode"] = "before"
        hints["anchor_terms_left"] = _extract_content_terms(phrase)

    return hints

def _anchor_spans_from_terms(
    video_actions_timeline: list[tuple[str, str, float, float]],
    terms: list[str],
) -> list[tuple[float, float]]:
    """Find action intervals whose names match anchor terms."""
    if not terms:
        return []
    spans: list[tuple[float, float]] = []
    for _code, name, s, e in video_actions_timeline:
        if _matched_terms_in_action_name(name, terms):
            spans.append((float(s), float(e)))
    return _merge_intervals_sorted(spans)

def _segment_temporal_relevance(
    *,
    start_sec: float,
    end_sec: float,
    mode: str | None,
    anchor_left: list[tuple[float, float]],
    anchor_right: list[tuple[float, float]],
) -> tuple[float, str]:
    """Temporal relevance score for one action segment under query temporal intent."""
    if mode is None:
        return 0.0, "none"

    s = float(start_sec)
    e = float(end_sec)

    if mode == "after" and anchor_left:
        ref_end = max(x[1] for x in anchor_left)
        if s >= ref_end:
            dist = s - ref_end
            return 8.0 / (1.0 + dist), "after_anchor"
        if e <= ref_end:
            return -4.0, "before_anchor"
        return 0.2, "crosses_anchor"

    if mode == "before" and anchor_left:
        ref_start = min(x[0] for x in anchor_left)
        if e <= ref_start:
            dist = ref_start - e
            return 8.0 / (1.0 + dist), "before_anchor"
        if s >= ref_start:
            return -4.0, "after_anchor"
        return 0.2, "crosses_anchor"

    if mode == "between" and anchor_left and anchor_right:
        left_end = max(x[1] for x in anchor_left)
        right_start = min(x[0] for x in anchor_right)
        if right_start <= left_end:
            return 0.0, "invalid_between_window"
        if s >= left_end and e <= right_start:
            center = 0.5 * (s + e)
            target = 0.5 * (left_end + right_start)
            dist = abs(center - target)
            return 8.0 / (1.0 + dist), "between_anchors"
        if e > left_end and s < right_start:
            return 0.8, "overlaps_between_window"
        return -4.0, "outside_between_window"

    if mode == "while" and anchor_left:
        overlaps = any(not (e <= a_s or s >= a_e) for a_s, a_e in anchor_left)
        if overlaps:
            return 3.0, "overlaps_anchor"
        nearest = min(min(abs(s - a_e), abs(a_s - e)) for a_s, a_e in anchor_left)
        return 1.0 / (1.0 + nearest), "near_anchor"

    return 0.0, "no_anchor_match"

def build_query_guided_action_lookup_order(
    *,
    question: str,
    video_actions_timeline: list[tuple[str, str, float, float]],
    noun_terms: list[str],
    verb_terms: list[str],
) -> list[dict]:
    """
    Build query-guided action lookup order:
    ranked action segments (name + [start, end]) based on query meaning + temporal intent.
    """
    if not video_actions_timeline:
        return []

    hints = _parse_temporal_query_hints(question)
    mode = hints["mode"]
    anchor_left_terms = list(hints["anchor_terms_left"])
    anchor_right_terms = list(hints["anchor_terms_right"])
    anchor_left_spans = _anchor_spans_from_terms(video_actions_timeline, anchor_left_terms)
    anchor_right_spans = _anchor_spans_from_terms(video_actions_timeline, anchor_right_terms)

    query_terms: list[str] = []
    seen_terms: set[str] = set()
    for t in noun_terms + verb_terms + _extract_content_terms(question):
        if t not in seen_terms:
            seen_terms.add(t)
            query_terms.append(t)
    anchor_term_set = set(anchor_left_terms + anchor_right_terms)
    if mode in {"after", "before", "between", "while"} and anchor_term_set:
        target_query_terms = [t for t in query_terms if t not in anchor_term_set]
    else:
        target_query_terms = list(query_terms)

    max_end = max(float(e) for _c, _n, _s, e in video_actions_timeline) if video_actions_timeline else 0.0

    scored: list[dict] = []
    for code, name, s, e in sorted(video_actions_timeline, key=lambda x: (x[2], x[3], x[0])):
        matched_noun_terms = _matched_terms_in_action_name(name, noun_terms)
        matched_verb_terms = _matched_terms_in_action_name(name, verb_terms)
        matched_query_terms = _matched_terms_in_action_name(name, target_query_terms)
        matched_anchor_terms = _matched_terms_in_action_name(
            name, anchor_left_terms + anchor_right_terms
        )

        lexical_score = (
            2.2 * len(matched_noun_terms)
            + 2.5 * len(matched_verb_terms)
            + 1.2 * len(matched_query_terms)
        )
        anchor_score = 0.4 * len(matched_anchor_terms)
        if mode in {"after", "before", "between"} and matched_anchor_terms:
            # Anchor actions are often temporal context, not the target answer action.
            anchor_score -= 2.5 * len(matched_anchor_terms)
        temporal_score, temporal_signal = _segment_temporal_relevance(
            start_sec=float(s),
            end_sec=float(e),
            mode=mode,
            anchor_left=anchor_left_spans,
            anchor_right=anchor_right_spans,
        )

        position_score = 0.0
        if max_end > 0:
            mid = 0.5 * (float(s) + float(e))
            if hints["prefer_first"] and not hints["prefer_last"]:
                position_score += 1.0 - (mid / max_end)
            if hints["prefer_last"] and not hints["prefer_first"]:
                position_score += mid / max_end

        total_score = lexical_score + anchor_score + temporal_score + position_score
        scored.append(
            {
                "code": code,
                "name": name,
                "start_sec": float(s),
                "end_sec": float(e),
                "score": float(total_score),
                "matched_noun_terms": matched_noun_terms,
                "matched_verb_terms": matched_verb_terms,
                "matched_query_terms": matched_query_terms,
                "target_query_terms": target_query_terms,
                "matched_anchor_terms": matched_anchor_terms,
                "temporal_signal": temporal_signal,
                "temporal_mode": mode,
            }
        )

    def _tie_time(item: dict) -> float:
        if mode == "before":
            return -float(item["end_sec"])
        if mode == "after":
            return float(item["start_sec"])
        if hints["prefer_last"] and not hints["prefer_first"]:
            return -float(item["start_sec"])
        return float(item["start_sec"])

    scored.sort(
        key=lambda d: (
            -float(d["score"]),
            _tie_time(d),
            str(d["code"]),
            str(d["name"]).lower(),
        )
    )

    ordered: list[dict] = []
    for i, d in enumerate(scored, start=1):
        ordered.append(
            {
                "lookup_rank": i,
                "code": str(d["code"]),
                "name": str(d["name"]),
                "start_sec": float(d["start_sec"]),
                "end_sec": float(d["end_sec"]),
                "relevance_score": round(float(d["score"]), 6),
            }
        )
    return ordered
