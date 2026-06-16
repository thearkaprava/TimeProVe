#!/usr/bin/env python3
"""Remove s2_vlm_desc intermediates once every question has a final answer."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path


def cleanup_s2_intermediates_if_complete(
    data_path: Path,
    final_out_dir: Path,
    s2_out_dir: Path,
    *,
    log: logging.Logger | None = None,
) -> bool:
    """
    Delete contents of ``s2_out_dir`` when each question in ``data_path`` has
  ``final_out_dir/<question_id>.json``. Returns True if cleanup ran.
    """
    logger = log or logging.getLogger(__name__)
    if not data_path.is_file():
        logger.warning("Skip s2 cleanup: data file missing: %s", data_path)
        return False
    raw = data_path.read_text(encoding="utf-8").strip()
    if not raw:
        logger.warning("Skip s2 cleanup: data file is empty: %s", data_path)
        return False
    data = json.loads(raw)
    if not isinstance(data, dict) or not data:
        logger.warning("Skip s2 cleanup: expected non-empty dict in %s", data_path)
        return False

    missing = [
        qid for qid in data if not (final_out_dir / f"{qid}.json").is_file()
    ]
    if missing:
        logger.info(
            "Skip s2 cleanup: %d/%d final answer(s) missing in %s",
            len(missing),
            len(data),
            final_out_dir,
        )
        return False

    if not s2_out_dir.is_dir():
        logger.info("Skip s2 cleanup: s2_out_dir does not exist: %s", s2_out_dir)
        return False

    removed = 0
    for child in s2_out_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    logger.info(
        "Cleaned s2 intermediates under %s (%d top-level entries removed; "
        "%d final answer(s) present)",
        s2_out_dir,
        removed,
        len(data),
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete s2_vlm_desc contents after all final answers exist."
    )
    parser.add_argument("--data", type=Path, required=True, help="Prepared questions JSON")
    parser.add_argument(
        "--final-out-dir",
        type=Path,
        required=True,
        help="Directory containing per-question final answer JSON files",
    )
    parser.add_argument(
        "--s2-out",
        type=Path,
        required=True,
        help="s2_vlm_desc directory whose contents should be removed",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO logging",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    ok = cleanup_s2_intermediates_if_complete(
        args.data,
        args.final_out_dir,
        args.s2_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
