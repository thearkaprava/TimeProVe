#!/usr/bin/env python3
"""Compute accuracy from a predictions JSON (answer vs ground_truth_answer).

Each prediction is long free-form text (e.g. description-based QA). Instead of parsing a
short label and string-matching to ``ground_truth_answer``, a local Ollama model judges
whether the written answer **supports** the reference label (paraphrase and entailment
allowed). See ``ground_truth_supported_by_description_ollama`` in ``llm_answer_parsing``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from llm_answer_parsing import ground_truth_supported_by_description_ollama


def _non_empty_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def prediction_text_from_record(record: dict) -> str | None:
    """Prefer ``answer``, then ``final_answer`` / nested shapes used in merged pipelines."""
    for key in ("answer", "final_answer", "raw_prediction"):
        v = record.get(key)
        if isinstance(v, dict):
            for inner in ("exact_answer", "answer", "output"):
                t = _non_empty_str(v.get(inner))
                if t is not None:
                    return t
        else:
            t = _non_empty_str(v)
            if t is not None:
                return t
    return _non_empty_str(record.get("final_exact_answer"))


def evaluate_records(
    records: list[dict], *, ollama_model: str
) -> tuple[list[dict], dict[str, float | int]]:
    """
    Returns (per_question, summary_metrics).

    Skipped rows have missing prediction/ground_truth_answer or a non-null error field.
    A row is correct when Ollama returns match (``True``); mismatch is wrong; unknown /
    unparseable (``None``) is counted wrong.
    """
    per_question: list[dict] = []
    n_correct = 0
    n_evaluated = 0
    n_skipped = 0

    for r in records:
        if not isinstance(r, dict):
            per_question.append({"invalid_record": True})
            n_skipped += 1
            continue
        qid = r.get("question_id")
        q = r.get("question", "")
        pred_raw = prediction_text_from_record(r)
        gt = r.get("ground_truth_answer")
        err = r.get("error")
        if err:
            per_question.append(
                {
                    "question_id": qid,
                    "ground_truth_answer": gt,
                    "raw_prediction": pred_raw,
                    "final_answer": None,
                    "skipped": True,
                }
            )
            n_skipped += 1
            continue
        if pred_raw is None or gt is None:
            per_question.append(
                {
                    "question_id": qid,
                    "ground_truth_answer": gt,
                    "raw_prediction": pred_raw,
                    "final_answer": None,
                    "skipped": True,
                }
            )
            n_skipped += 1
            continue

        supported = ground_truth_supported_by_description_ollama(
            str(q), pred_raw, str(gt), model=ollama_model
        )
        n_evaluated += 1
        final_answer = "match" if supported is True else "mismatch" if supported is False else None
        per_question.append(
            {
                "question_id": qid,
                "ground_truth_answer": gt,
                "raw_prediction": pred_raw,
                "final_answer": final_answer,
            }
        )
        if supported is True:
            n_correct += 1

    summary: dict[str, float | int] = {
        "accuracy": (n_correct / n_evaluated) if n_evaluated else 0.0,
        "correct": n_correct,
        "evaluated": n_evaluated,
        "skipped": n_skipped,
    }
    return per_question, summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="JSON array of dicts with ground_truth_answer and answer or final_answer",
    )
    p.add_argument(
        "--ollama-model",
        default="llama3.1",
        help="Ollama model that judges support for the reference label (default: llama3.1)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional output path for final_results-style JSON.",
    )
    args = p.parse_args(argv)

    data = json.loads(args.input.read_text())
    if not isinstance(data, list):
        print("Expected a JSON array of records.", file=sys.stderr)
        return 1

    per_question, summary = evaluate_records(
        data, ollama_model=args.ollama_model
    )
    output_payload = {
        "per_question": per_question,
        "accuracies": {
            "final_answer": summary,
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_payload, indent=2))

    print(
        f"accuracy: {float(summary['accuracy']):.6f} "
        f"({int(summary['correct'])}/{int(summary['evaluated'])})"
    )
    skipped = int(summary["skipped"])
    if skipped:
        print(f"skipped: {skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
