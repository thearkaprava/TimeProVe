#!/usr/bin/env python3
"""Merge top-k per-question outputs into one JSON array for evaluation."""

from __future__ import annotations

import argparse
import json
import os.path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional


def _non_empty_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_final_answer(obj: Dict[str, Any]) -> Optional[str]:
    """Resolve final answer from supported top-k schemas."""
    final_answer = obj.get("final_answer")
    if isinstance(final_answer, dict):
        for key in ("exact_answer", "answer", "output"):
            ans = _non_empty_str(final_answer.get(key))
            if ans is not None:
                return ans
    else:
        ans = _non_empty_str(final_answer)
        if ans is not None:
            return ans

    majority_vote = obj.get("majority_vote")
    if isinstance(majority_vote, dict):
        ans = _non_empty_str(majority_vote.get("final_answer"))
        if ans is not None:
            return ans

    for key in ("final_exact_answer", "answer"):
        ans = _non_empty_str(obj.get(key))
        if ans is not None:
            return ans
    return None


def _row_from_final_json(obj: Dict[str, Any], video_root: str) -> Dict[str, Any]:
    qid = obj["question_id"]
    video_id = obj.get("video_id") or str(qid).split("-", 1)[0]
    video_name = f"{video_id}.mp4"
    video_path = osp.join(video_root, video_name)
    question = obj.get("query") or obj.get("question") or ""
    gt = obj.get("ground_truth_answer", "")

    return {
        "question_id": qid,
        "question": question,
        "video_name": video_name,
        "video_path": video_path,
        "ground_truth_answer": gt,
        "final_answer": _extract_final_answer(obj),
        "top_k_actions": obj.get("top_k_actions"),
    }


def merge_final_ans_dir(input_dir: str | Path, video_root: str) -> List[Dict[str, Any]]:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No .json files found in input directory: {input_path}")

    rows: List[Dict[str, Any]] = []
    for path in json_files:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected a JSON object in {path}, got {type(obj)}")
        rows.append(_row_from_final_json(obj, video_root))

    rows.sort(key=lambda r: r["question_id"])
    return rows


def main() -> None:
    default_in = (
        "/home/asinha13/projects/Interpretable_VLMA3/workdirs/"
        "otb_otb_gtquery_actnprop_desc/s3_llm_final_ans"
    )
    default_video_root = "/data/vidlab_datasets/smarthome/untrimmed/Videos_mp4"
    parser = argparse.ArgumentParser(
        description=(
            "Merge top-k s3_llm_final_ans per-question JSON files into one array with "
            "one record per question using each record's final answer."
        )
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=default_in,
        help="Directory containing one JSON per question (e.g. s3_llm_final_ans).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for merged JSON (array of records).",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default=default_video_root,
        help="OTB video root; video_path is <root>/<video_id>.mp4.",
    )
    args = parser.parse_args()

    merged = merge_final_ans_dir(args.input_dir, args.video_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {len(merged)} records to {out_path}")


if __name__ == "__main__":
    main()
