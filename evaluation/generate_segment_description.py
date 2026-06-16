#!/usr/bin/env python3
"""
Generate a three-sentence video description per file in a directory (actions, objects,
human–object interactions) and save JSON keyed by video filename, with tags parsed from
each name (segment / source hints). Uses the same VideoLLaMA3 inference path as
evaluation/evaluate.py (evaluation.register.INFERENCES + mm_infer).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import random
import sys
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(".")
from evaluation.register import INFERENCES
from videollama3 import disable_torch_init


DESCRIPTION_PROMPT = (
    "Write exactly three detailed sentences describing this video clip. "
    "Focus on: (1) what actions occur, especially by any people; (2) the main objects "
    "and scene elements visible; (3) how people interact with objects (handling, using, "
    "moving, or contacting them). Be concrete and specific. Use exactly three sentences."
)


PROMPT_STYLE_GENERIC = "generic"
PROMPT_STYLE_QUERY_SOFT = "query_soft"
PROMPT_STYLE_QUERY_STRICT = "query_strict"
PROMPT_STYLE_QUERY_TVG = "query_tvg"
PROMPT_STYLE_CHOICES = (
    PROMPT_STYLE_GENERIC,
    PROMPT_STYLE_QUERY_SOFT,
    PROMPT_STYLE_QUERY_STRICT,
    PROMPT_STYLE_QUERY_TVG,
)


def build_clip_description_prompt(
    question: Optional[str],
    prompt_style: str = PROMPT_STYLE_QUERY_SOFT,
    *,
    action_context: Optional[str] = None,
) -> str:
    """
    Text sent to the VLM after the video. When ``question`` is set, describe
    evidence in the clip that is most useful for answering it.

    ``action_context`` is an optional text hint for the VLM (e.g. a tentative
    answer inferred only from detected action labels and timestamps, or a short
    action-label summary). When provided, it is appended so the model can weigh
    that guidance against what it actually sees in the clip.
    """
    style = str(prompt_style or PROMPT_STYLE_QUERY_SOFT).strip().lower()
    q = (question or "").strip()
    hint = (action_context or "").strip()
    if style == PROMPT_STYLE_GENERIC or not q:
        if hint:
            return (
                DESCRIPTION_PROMPT
                + "\n\nTentative guidance (from action labels / times or a text-only model, not from pixels):\n"
                f"{hint}\n"
                "Use this only as a hypothesis to notice relevant details; trust the video if they disagree."
            )
        return DESCRIPTION_PROMPT
    if style == PROMPT_STYLE_QUERY_STRICT:
        parts: list[str] = [
            "You will see a video clip and a question about the video.",
            "",
            f'Question:\n"{q}"',
        ]
        if hint:
            parts += [
                "",
                "Tentative guidance (text-only model from detected actions and timestamps, not from video pixels):",
                hint,
                "Treat this as a hypothesis about what matters for the question; verify everything against the clip.",
            ]
        parts += [
            "",
            "Write exactly three sentences describing ONLY what is visible in this clip.",
            "Prefer details that help answer the question: actions, involved objects, spatial relations, and event order.",
            "If the clip does not show enough evidence for the question, still describe the clip faithfully and note that "
            "the answer cannot be determined from this clip alone.",
            "Do not provide the final answer directly; provide an evidence-focused description.",
        ]
        return "\n".join(parts)
    if style == PROMPT_STYLE_QUERY_TVG:
        parts = [
            "You will see one video clip: a candidate temporal window from automatic action detection.",
            "Your description will be used to verify whether a grounding query (or parts of it) appears in this clip.",
            "",
            f'Grounding query:\n"{q}"',
        ]
        if hint:
            parts += [
                "",
                "MS-TEMBA detected actions for this window (labels and times; not from pixels):",
                hint,
                "Use only as a hint for what to look for; trust the video if they disagree.",
            ]
        parts += [
            "",
            "Before writing, mentally break the query into key visual parts: who (actor), main action/verb, "
            "which object(s)/place, and any relations. A separate strict verifier will require the main action "
            "to be supported; your job is to report evidence faithfully, including partial visibility.",
            "",
            "Write exactly three sentences, grounded only in visible pixels:",
            "(1) Main actions and who performs them.",
            "(2) Key objects, location, and scene context.",
            "(3) Query presence check: for each key part, say if it is visible, not visible, or unclear. "
            "Explicitly state whether the query's main action/event is visible. "
            "Use synonyms when the query word does not appear literally (e.g. 'holds' for 'grasping'). "
            "If only part of the query is shown, name which parts match and which are missing; do not require "
            "every word of the query to appear on screen.",
            "Do not output timestamps or a standalone yes/no answer; put the presence check in sentence (3).",
        ]
        return "\n".join(parts)
    parts: list[str] = [
        "You will see a video clip and a question about the full video.",
        "",
        f'Question:\n"{q}"',
    ]
    if hint:
        parts += [
            "",
            "Tentative guidance (text-only model from detected actions and timestamps, not from video pixels):",
            hint,
            "Use it to prioritize details; still describe the clip faithfully and do not treat it as ground truth.",
        ]
    parts += [
        "",
        "Write exactly three detailed sentences describing what is visible in this clip.",
        "Keep the description generally useful and grounded in visible content: actions, people, objects, scene context, and temporal order.",
        "Use the question only as a soft focus to prioritize relevant details, but do not ignore other important visible events.",
        "Do not guess unseen events and do not answer the question directly.",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate 3-sentence segment descriptions for videos in a directory."
    )
    p.add_argument("--video-dir", "--video_dir", type=str, required=True, help="Directory containing videos.")
    p.add_argument("--model-path", "--model_path", type=str, required=True)
    p.add_argument(
        "--output-json",
        "--output_json",
        type=str,
        required=True,
        help="Path to write JSON (descriptions keyed by video basename).",
    )
    p.add_argument("--glob", type=str, default="*.mp4", help="Glob pattern under video-dir (default: *.mp4).")
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--max-frames", "--max_frames", type=int, default=180)
    p.add_argument("--max-visual-tokens", "--max_visual_tokens", type=int, default=None)
    p.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--prompt-style",
        "--prompt_style",
        type=str,
        default=PROMPT_STYLE_QUERY_SOFT,
        choices=PROMPT_STYLE_CHOICES,
        help="Description prompt style: generic | query_soft | query_strict.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device map for the model, e.g. "cuda:0".',
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume",
        action="store_true",
        help="If output JSON exists, skip videos already present in it.",
    )
    p.add_argument(
        "--tag-delimiter",
        "--tag_delimiter",
        type=str,
        default="_",
        help="Split video stem on this string to build tags (default: underscore).",
    )
    return p.parse_args()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_videos(video_dir: str, pattern: str) -> List[str]:
    search = osp.join(video_dir, "**", pattern) if "**" not in pattern else osp.join(video_dir, pattern)
    paths = sorted(glob.glob(search, recursive=True))
    if not paths:
        alt = osp.join(video_dir, pattern)
        paths = sorted(glob.glob(alt))
    if not paths:
        raise FileNotFoundError(f"No videos matching {pattern!r} under {video_dir!r}")
    return paths


def tags_from_basename(basename: str, delimiter: str) -> Dict[str, Any]:
    stem, _ = osp.splitext(basename)
    if not delimiter:
        parts = [stem] if stem else []
    else:
        parts = [p for p in stem.split(delimiter) if p]
        if not parts and stem:
            parts = [stem]
    segment_tags = parts[1:] if len(parts) > 1 else []
    source_id: Optional[str] = parts[0] if parts else None
    return {
        "tags": parts,
        "segment_tags": segment_tags,
        "source_id": source_id,
    }


def run_one_video(
    video_path: str,
    model,
    processor,
    mm_infer_fn,
    fps: int,
    max_frames: int,
    max_new_tokens: int,
    *,
    question: Optional[str] = None,
    prompt_style: str = PROMPT_STYLE_QUERY_SOFT,
    action_context: Optional[str] = None,
) -> str:
    frames, timestamps = processor.load_video(
        video_path,
        start_time=None,
        end_time=None,
        precise_time=True,
        fps=fps,
        max_frames=max_frames,
    )
    image_inputs = processor.process_images([frames], merge_size=2, return_tensors="pt")
    text_prompt = build_clip_description_prompt(
        question, prompt_style=prompt_style, action_context=action_context
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
    response = mm_infer_fn(
        data_dict,
        model=model,
        tokenizer=processor.tokenizer,
        modal="video",
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    return response.strip()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    disable_torch_init()

    model_init, mm_infer = INFERENCES(args.model_path)
    model, processor = model_init(
        args.model_path,
        args.max_visual_tokens,
        device_map={"": args.device},
    )

    video_paths = discover_videos(args.video_dir, args.glob)
    out_path = osp.abspath(args.output_json)
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)

    payload: Dict[str, Any] = {
        "model_path": args.model_path,
        "video_dir": osp.abspath(args.video_dir),
        "glob": args.glob,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "prompt_style": args.prompt_style,
        "descriptions": {},
    }

    if args.resume and osp.isfile(out_path):
        try:
            with open(out_path, "r") as f:
                existing = json.load(f)
            if isinstance(existing.get("descriptions"), dict):
                payload["descriptions"] = existing["descriptions"]
        except (json.JSONDecodeError, OSError):
            pass

    for video_path in tqdm(video_paths, desc="Videos"):
        basename = osp.basename(video_path)
        if args.resume and basename in payload["descriptions"]:
            entry = payload["descriptions"][basename]
            if isinstance(entry, dict) and entry.get("description") and not str(entry["description"]).startswith(
                "ERROR:"
            ):
                continue

        meta = tags_from_basename(basename, args.tag_delimiter)
        try:
            description = run_one_video(
                video_path,
                model,
                processor,
                mm_infer,
                args.fps,
                args.max_frames,
                args.max_new_tokens,
                prompt_style=args.prompt_style,
            )
        except Exception:
            traceback.print_exc()
            description = f"ERROR: {traceback.format_exc()}"

        payload["descriptions"][basename] = {
            "video_path": osp.abspath(video_path),
            "description": description,
            **meta,
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

    del model
    torch.cuda.empty_cache()
    print(f"Wrote {len(payload['descriptions'])} entries to {out_path}")


if __name__ == "__main__":
    main()
