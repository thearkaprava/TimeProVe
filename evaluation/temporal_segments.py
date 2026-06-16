#!/usr/bin/env python3
"""
Load OTB action-detection scores (C x T, probabilities per class per temporal bin),
find bins above a threshold for selected class ids, log them, and cut the source video.

Actions that pass the threshold are **sorted** by earliest bin then action id. All bins
from those actions are **unioned**; **contiguous** bin ranges form one segment each; a
**gap** (e.g. bins 12–15 vs 17–20) yields **separate output videos** (no concatenation).

When there is only one contiguous segment (including the no-threshold fallback that
exports a single bin), the bin range is expanded by ``--bin-context-pad`` bins on each
side (default 1), so a one-bin selection becomes three bins (context before and after).

Bins map to decoded **video frame indices**: for N frames and T bins, bin t covers
frames [floor(t*N/T), floor((t+1)*N/T)); time(frame) uses stream r_frame_rate. Cuts use
ffmpeg trim on frame numbers.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


def get_format_duration_seconds(video_path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def get_video_stream_meta(video_path: Path) -> dict:
    """nb_frames, stream duration, and r_frame_rate num/den for the first video stream."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,duration,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream in {video_path}")
    s = streams[0]
    rf = s.get("r_frame_rate") or "0/1"
    num_s, den_s = rf.split("/")
    fps_num, fps_den = int(num_s), int(den_s)
    if fps_den == 0:
        raise RuntimeError(f"Invalid r_frame_rate {rf!r}")
    nb_frames = s.get("nb_frames")
    if nb_frames is not None:
        nb_frames = int(nb_frames)
    dur = s.get("duration")
    stream_duration = float(dur) if dur is not None else None
    return {
        "nb_frames": nb_frames,
        "stream_duration": stream_duration,
        "fps_num": fps_num,
        "fps_den": fps_den,
    }


def count_video_frames(video_path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def contiguous_runs(sorted_indices: list[int]) -> list[tuple[int, int]]:
    """Inclusive segment index ranges for contiguous indices."""
    if not sorted_indices:
        return []
    runs: list[tuple[int, int]] = []
    a = b = sorted_indices[0]
    for x in sorted_indices[1:]:
        if x == b + 1:
            b = x
        else:
            runs.append((a, b))
            a = b = x
    runs.append((a, b))
    return runs


def pad_bin_run_single_segment(
    rs: int,
    re: int,
    num_bins: int,
    pad_bins: int,
) -> tuple[int, int]:
    """
    Expand inclusive bin range [rs, re] by pad_bins on each side, clamped to [0, num_bins - 1].
    Used when there is only one contiguous segment so the clip includes temporal context.
    """
    if pad_bins <= 0:
        return rs, re
    lo = max(0, rs - pad_bins)
    hi = min(num_bins - 1, re + pad_bins)
    return lo, hi


def bin_run_to_frame_range(
    run_start: int, run_end: int, num_bins: int, nb_frames: int
) -> tuple[int, int]:
    """
    Map inclusive bin indices [run_start, run_end] to half-open frame indices
    [start_frame, end_frame) over nb_frames decoded frames (same as uniform binning
    in frame space).
    """
    start_f = (run_start * nb_frames) // num_bins
    end_f = ((run_end + 1) * nb_frames) // num_bins
    end_f = min(nb_frames, max(end_f, start_f + 1))
    start_f = min(max(0, start_f), nb_frames - 1)
    return start_f, end_f


def frame_range_to_time_sec(
    start_f: int, end_f_excl: int, fps_num: int, fps_den: int
) -> tuple[float, float]:
    """Half-open [start_f, end_f_excl) in frames -> [t0, t1) seconds (stream time base)."""
    t0 = start_f * fps_den / fps_num
    t1 = end_f_excl * fps_den / fps_num
    return t0, t1


@functools.lru_cache(maxsize=1)
def ffmpeg_reencode_codec_args() -> tuple[str, ...]:
    """
    Video+audio encode flags that work across ffmpeg builds.

    Conda ffmpeg 4.3 often ships libopenh264 without a libx264 build that accepts ``-crf``.
    """
    audio = ("-c:a", "aac", "-b:a", "128k")
    candidates: list[tuple[str, ...]] = [
        ("-c:v", "libx264", "-crf", "20", "-preset", "fast"),
        ("-c:v", "libx264", "-preset", "fast", "-qp", "23"),
        ("-c:v", "libopenh264", "-b:v", "1500k"),
        ("-c:v", "mpeg4", "-qscale:v", "3"),
    ]
    probe = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=16x16:d=0.1",
    ]
    for enc in candidates:
        cmd = probe + list(enc) + ["-f", "null", os.devnull]
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            logging.debug("ffmpeg reencode using: %s", " ".join(enc))
            return enc + audio
    logging.warning("ffmpeg encoder probe failed; falling back to mpeg4 qscale")
    return ("-c:v", "mpeg4", "-qscale:v", "3") + audio


def ffmpeg_extract_clip_frames(
    video_in: Path,
    start_frame: int,
    end_frame_excl: int,
    fps_num: int,
    fps_den: int,
    video_out: Path,
    reencode: bool = True,
) -> None:
    """
    Frame-accurate extract: video trim by frame index; audio atrim by matching seconds.
    end_frame_excl is the first frame not included (ffmpeg trim end_frame semantics).
    """
    if end_frame_excl <= start_frame:
        raise ValueError(f"Empty frame range: [{start_frame}, {end_frame_excl})")
    video_out.parent.mkdir(parents=True, exist_ok=True)
    t0, t1 = frame_range_to_time_sec(start_frame, end_frame_excl, fps_num, fps_den)
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame_excl},setpts=PTS-STARTPTS"
    af = f"atrim=start={t0}:end={t1},asetpts=PTS-STARTPTS"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_in),
        "-vf",
        vf,
        "-af",
        af,
    ]
    if reencode:
        cmd += list(ffmpeg_reencode_codec_args())
    else:
        cmd += ["-c", "copy"]
    cmd += [str(video_out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{r.stderr}")


def load_scores(pkl_path: Path, video_key: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if video_key not in data:
        keys_sample = list(data.keys())[:10]
        raise KeyError(f"Missing key {video_key!r} in pickle. Sample keys: {keys_sample}")
    scores = np.asarray(data[video_key], dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D array for {video_key}, got shape {scores.shape}")
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("/data/vidlab_datasets/smarthome/untrimmed/Videos_mp4/P01A01.mp4"),
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=Path(
            "/home/asinha13/projects/TimeProVe/data/TSU_best_AD.pkl"
        ),
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default=None,
        help="Key in the pickle (default: stem of --video, e.g. YSKX3)",
    )
    parser.add_argument(
        "--action-ids",
        type=int,
        nargs="+",
        default=[70, 71, 72, 73, 74, 75, 76, 79, 80],
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file path (default: next to --out with .log suffix)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for extracted mp4 clips (default: ./<video_key>_ad_segments/)",
    )
    parser.add_argument(
        "--no-fallback-max-segment",
        action="store_true",
        help="If no bin exceeds the threshold, exit with error instead of exporting "
        "the single highest-probability bin among selected action ids.",
    )
    parser.add_argument(
        "--bin-context-pad",
        type=int,
        default=1,
        help="When there is only one contiguous segment, expand its bin range by this many "
        "bins on each side (default: 1 → one bin before + one bin after; set 0 to disable). "
        "Also applies to the no-threshold fallback (single bin).",
    )
    args = parser.parse_args()

    video_key = args.video_key or args.video.stem
    out_dir = args.out or (Path.cwd() / f"{video_key}_ad_segments")
    out_dir = out_dir.resolve()
    log_path = args.log or (out_dir / f"{video_key}_ad_segments.log")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("extract_temporal_segments")

    scores = load_scores(args.pkl, video_key)
    num_classes, num_bins = scores.shape
    fmt_dur = get_format_duration_seconds(args.video)
    vmeta = get_video_stream_meta(args.video)
    nb_frames = vmeta["nb_frames"]
    if nb_frames is None:
        log.warning("nb_frames missing from probe; counting frames (may be slow)")
        nb_frames = count_video_frames(args.video)
    fps_num, fps_den = vmeta["fps_num"], vmeta["fps_den"]
    stream_dur = vmeta["stream_duration"]
    log.info(
        "video=%s format_duration_sec=%.6f stream_duration_sec=%s nb_frames=%d r_frame_rate=%d/%d",
        args.video,
        fmt_dur,
        f"{stream_dur:.6f}" if stream_dur is not None else "n/a",
        nb_frames,
        fps_num,
        fps_den,
    )
    log.info("pkl key=%s scores shape (C,T)=(%d,%d)", video_key, num_classes, num_bins)

    for aid in args.action_ids:
        if aid < 0 or aid >= num_classes:
            log.error("action id %d out of range [0, %d)", aid, num_classes)
            return 1

    # Log every temporal bin above threshold at least once per action
    any_above: dict[int, np.ndarray] = {}
    for aid in args.action_ids:
        row = scores[aid]
        above = np.where(row > args.threshold)[0]
        any_above[aid] = above
        if len(above):
            vals = row[above]
            log.info(
                "action_id=%d bins_above_threshold=%s probs=%s",
                aid,
                above.tolist(),
                np.round(vals, 6).tolist(),
            )
        else:
            log.info(
                "action_id=%d no bin above threshold (max prob=%.6f at bin %d)",
                aid,
                float(row.max()),
                int(row.argmax()),
            )

    actions_with_hits = [aid for aid in args.action_ids if len(any_above[aid]) > 0]
    use_fallback = len(actions_with_hits) == 0 and not args.no_fallback_max_segment

    if len(actions_with_hits) > 0:
        actions_sorted = sorted(
            actions_with_hits,
            key=lambda a: (int(any_above[a].min()), a),
        )
        log.info(
            "Actions with threshold hits (sorted by min bin index, then action id): %s",
            actions_sorted,
        )
        union_bins: set[int] = set()
        for aid in actions_sorted:
            union_bins.update(int(x) for x in any_above[aid].tolist())
        sorted_bins = sorted(union_bins)
        runs = contiguous_runs(sorted_bins)
        log.info("Union of bin indices across those actions: %s", sorted_bins)
        log.info(
            "Distinct contiguous bin runs (each becomes one mp4 if gaps exist): %s",
            runs,
        )
    elif use_fallback:
        best_val = -1.0
        best_aid = args.action_ids[0]
        best_t = 0
        for aid in args.action_ids:
            row = scores[aid]
            t = int(row.argmax())
            v = float(row[t])
            if v > best_val:
                best_val = v
                best_aid = aid
                best_t = t
        actions_sorted = [best_aid]
        runs = [(best_t, best_t)]
        log.warning(
            "No selected action exceeded threshold=%.4f; exporting fallback single bin "
            "with max prob among selected: action_id=%d bin=%d prob=%.6f",
            args.threshold,
            best_aid,
            best_t,
            best_val,
        )
    else:
        log.error(
            "No temporal bin above threshold=%.4f for any selected action; "
            "omit --no-fallback-max-segment to export a max-prob demo bin.",
            args.threshold,
        )
        return 1

    written: list[Path] = []
    single_segment = len(runs) == 1
    for seg_i, (rs, re) in enumerate(runs):
        rs_eff, re_eff = rs, re
        if single_segment and args.bin_context_pad > 0:
            rs_eff, re_eff = pad_bin_run_single_segment(
                rs, re, num_bins, args.bin_context_pad
            )
            if (rs_eff, re_eff) != (rs, re):
                log.info(
                    "single segment: padded bin range [%d,%d] -> [%d,%d] (±%d bin(s) context; "
                    "%d bin(s) wide -> %d)",
                    rs,
                    re,
                    rs_eff,
                    re_eff,
                    args.bin_context_pad,
                    re - rs + 1,
                    re_eff - rs_eff + 1,
                )
        sf, ef = bin_run_to_frame_range(rs_eff, re_eff, num_bins, nb_frames)
        t0, t1 = frame_range_to_time_sec(sf, ef, fps_num, fps_den)
        log.info(
            "segment %d bins [%d,%d] -> frames [%d,%d) -> time [%.6f, %.6f) sec",
            seg_i,
            rs_eff,
            re_eff,
            sf,
            ef,
            t0,
            t1,
        )
        clip_path = out_dir / f"{video_key}_seg{seg_i:02d}_bins{rs_eff:02d}-{re_eff:02d}.mp4"
        ffmpeg_extract_clip_frames(args.video, sf, ef, fps_num, fps_den, clip_path)
        written.append(clip_path)

    for p in written:
        log.info("Wrote %s", p.resolve())
    log.info("Output directory %s", out_dir)
    log.info("Log file %s", log_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
