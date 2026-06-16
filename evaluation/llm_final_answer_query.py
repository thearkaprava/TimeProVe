#!/usr/bin/env python3
"""
Text-only inference with the repo's VideoLLaMA3 stack (no images/video).
Loads the model the same way as evaluation/evaluate.py (INFERENCES + model_init),
builds inputs like videollama3/infer.py (conversation -> processor), and writes
instruction, input, output, and generation settings to a JSON file.

Primary mode (--descriptions-json): loads a JSON file (e.g. segment_descriptions.json),
collects every `description` field, and answers --input / --input-file using those texts
as context.

OTB mode (no --descriptions-json): unless --instruction is set, the system prompt
is built from the OTB action-list file. After inference, class codes from the model
are merged with a fallback unless --no-otb-fallback.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# This file lives in <repo>/evaluation/; repo root is one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CLASSES_FILE = _REPO_ROOT / "data" / "TSU_Action_list.txt"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Lines look like: c012 Tidying up a table
_CLASS_LINE_RE = re.compile(r"^(c\d{3})\s+(.+)$")

_MIN_QUERY_TOKEN_LEN = 4
# Skip query tokens whose label match count exceeds this (avoids "holding", "someone", …).
_MAX_LABEL_FANOUT = 15

_STOPWORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "were",
        "they",
        "them",
        "what",
        "when",
        "where",
        "which",
        "while",
        "there",
        "their",
        "have",
        "been",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "about",
        "into",
        "than",
        "then",
        "some",
        "such",
        "only",
        "also",
        "just",
        "like",
        "here",
        "very",
        "after",
        "before",
        "being",
        "each",
        "same",
        "other",
        "both",
        "most",
        "many",
        "any",
        "interacting",
        "something",
        "someone",
        "anything",
        "using",
        "doing",
        "making",
        "during",
        "without",
        "within",
        "person",
        "people",
        "video",
        "scene",
    }
)


def _word_in_label(word: str, label: str) -> bool:
    """Whole-token match so 'book' does not match inside 'notebook'."""
    return (
        re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", label.lower()) is not None
    )


def _keyword_variants(w: str) -> list[str]:
    w = w.lower().strip()
    out = [w]
    if w.endswith("s") and len(w) > _MIN_QUERY_TOKEN_LEN:
        out.append(w[:-1])
    elif not w.endswith("s"):
        out.append(w + "s")
    return list(dict.fromkeys(out))


def fallback_classes_for_query_objects(
    query: str, classes: list[tuple[str, str, str]]
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """
    For each query token (length/stopword filtered), if the token matches whole words in
    relatively few class names, include every matching class. Returns (rows, keywords_used).
    """
    tokens = re.findall(r"[a-z][a-z0-9]*(?:/[a-z0-9]+)?s?", query.lower())
    seen: dict[str, tuple[str, str, str]] = {}
    keywords_used: list[str] = []
    for raw in tokens:
        if len(raw) < _MIN_QUERY_TOKEN_LEN or raw in _STOPWORDS:
            continue
        matches: list[tuple[str, str, str]] = []
        for kw in _keyword_variants(raw):
            mrows = [(c, n, name) for c, n, name in classes if _word_in_label(kw, name)]
            if mrows:
                matches = mrows
                break
        if not matches:
            continue
        if len(matches) > _MAX_LABEL_FANOUT:
            continue
        keywords_used.append(raw)
        for row in matches:
            seen[row[0]] = row
    rows = sorted(seen.values(), key=lambda t: int(t[1], 10))
    return rows, list(dict.fromkeys(keywords_used))


def parse_llm_class_codes(text: str, valid_codes: set[str]) -> set[str]:
    """Extract cNNN codes from lines that look like the required output format."""
    found: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^\s*(c\d{3})\s*[—\-]\s*\d+\s*[—\-]\s*", line.strip())
        if m and m.group(1) in valid_codes:
            found.add(m.group(1))
    return found


def merge_otb_llm_and_fallback(
    llm_text: str, query: str, classes: list[tuple[str, str, str]]
) -> tuple[str, dict[str, object]]:
    """
    Union of (1) classes parsed from LLM output and (2) fallback rows from query tokens.
    Returns formatted text and metadata for JSON.
    """
    by_code = {c: (c, n, name) for c, n, name in classes}
    valid = set(by_code)
    codes_llm = parse_llm_class_codes(llm_text, valid)
    fb_rows, fb_keywords = fallback_classes_for_query_objects(query, classes)
    codes_fb = {r[0] for r in fb_rows}
    all_codes = codes_llm | codes_fb
    if not all_codes:
        return llm_text, {"merged": False, "codes_from_llm": [], "codes_from_fallback": [], "keywords": []}

    merged = sorted((by_code[c] for c in all_codes if c in by_code), key=lambda t: int(t[1], 10))
    body = "\n".join(f"{c} — {n} — {name}" for c, n, name in merged)
    if fb_keywords:
        text_out = f"{', '.join(fb_keywords)}\n\n{body}"
    else:
        text_out = body
    meta: dict[str, object] = {
        "merged": True,
        "codes_from_llm": sorted(codes_llm, key=lambda x: int(x[1:], 10)),
        "codes_from_fallback": sorted(codes_fb - codes_llm, key=lambda x: int(x[1:], 10)),
        "keywords": fb_keywords,
    }
    return text_out, meta


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


def _recursive_collect_descriptions(obj: object, out: list[str]) -> None:
    """Collect string values for any key named 'description' (nested dicts/lists)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "description" and isinstance(v, str) and v.strip():
                out.append(v.strip())
            else:
                _recursive_collect_descriptions(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _recursive_collect_descriptions(item, out)


def load_descriptions_json(path: Path) -> tuple[list[str], list[str]]:
    """
    Load segment descriptions from JSON. Prefers top-level ``descriptions`` map
    (sorted keys) when present; otherwise collects all non-empty ``description`` strings
    in document order via recursion. Returns (descriptions, segment_keys_or_empty).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    texts: list[str] = []
    keys: list[str] = []

    desc_map = data.get("descriptions") if isinstance(data, dict) else None
    if isinstance(desc_map, dict):
        for seg_key in sorted(desc_map.keys()):
            item = desc_map[seg_key]
            if not isinstance(item, dict):
                continue
            d = item.get("description")
            if isinstance(d, str) and d.strip():
                keys.append(seg_key)
                texts.append(d.strip())
    if not texts:
        _recursive_collect_descriptions(data, texts)
        keys = []

    return texts, keys


def build_segment_qa_system_instruction() -> str:
    return (
        "You answer questions about a video using only the segment descriptions provided "
        "by the user. Base your answer on those descriptions; cite which segments support "
        "your reasoning when helpful (e.g. by segment index or label if given). "
        "If the descriptions do not contain enough information to answer, say so clearly "
        "and give the best partial answer or inference you can justify from the text."
    )


def build_segment_qa_user_content(
    descriptions: list[str],
    query: str,
    segment_keys: list[str],
) -> str:
    lines: list[str] = [
        "Below are descriptions of video segments (in order).",
        "",
    ]
    for i, text in enumerate(descriptions, start=1):
        if segment_keys and i - 1 < len(segment_keys):
            label = segment_keys[i - 1]
            lines.append(f"Segment {i} ({label}):")
        else:
            lines.append(f"Segment {i}:")
        lines.append(text)
        lines.append("")
    lines.extend(["Question:", query.strip()])
    return "\n".join(lines).strip()


def build_otb_instruction(classes: list[tuple[str, str, str]]) -> str:
    lines = [
        "You label the user's question with OTB activity classes from the list below. Follow the steps **exactly**.",
        "",
        "Step 1 — Objects in the question:",
        "Extract only **nouns / object phrases that appear in the question** (things, materials, food, furniture, etc.). Do not invent extra objects.",
        "",
        "Step 2 — Split compound questions:",
        "If the question has several parts (e.g. joined by \"while\", \"and\", or commas), treat **each part separately** when deciding how wide the class set is for each object.",
        "",
        "Step 3 — Broad vs narrow (critical):",
        "- **Broad** about an object: the text uses generic relation words such as *interacting with*, *using*, *handling*, *touching*, *with a [object]* without naming one specific verb of action. Then you must output **every** class in the list below whose **name** contains that object word (case-insensitive), including labels with suffixes like \"/s\". Scan from the first line to the last; **do not** stop after one hit.",
        "- **Narrow** about an object: the text names one concrete action for that object (e.g. *holding a …*, *throwing a …*, *eating a …*) that matches a single label closely. Then output **only** the class line(s) that match that wording and still mention the object.",
        "- **Do not** default to the single line that starts with \"Holding a …\" unless the question is narrow and only describes holding for that object. If the question is broad for that object, you must list **all** verbs that appear in the list for that object, not only holding.",
        "",
        "Step 4 — Combine:",
        "Take the union of classes from all parts; remove duplicates. Only include classes for objects that actually appear in the question.",
        "",
        "Step 5 — Output (required format):",
        "Line 1: comma-separated object words you used (from the question).",
        "Then one line per selected class, **only** in this form (copy from the list; same em dashes and spaces):",
        "`class_code — numeric_id — name`",
        "where `class_code` is three digits after `c` (e.g. c070), `numeric_id` is the same number **without** the `c` and **without** leading zeros (c070 → 70, c000 → 0), and `name` is the exact text after the second em dash in the list.",
        "No JSON, no bullet-only answers, no paraphrased names.",
        "",
        "Step 6 — Verify before sending:",
        "For each object that had **broad** wording in its clause, count how many list lines contain that object in the name; your answer must include that many lines for that object (unless Step 3 narrow applies).",
        "",
        "OTB classes (class_code — numeric_id — name):",
    ]
    for code, num, name in classes:
        lines.append(f"  {code} — {num} — {name}")
    return "\n".join(lines)


def build_conversation(instruction: str | None, user_text: str) -> list[dict]:
    conv = []
    if instruction and instruction.strip():
        conv.append({"role": "system", "content": instruction.strip()})
    conv.append({"role": "user", "content": user_text.strip()})
    return conv


def main() -> None:
    # Lazy import avoids pulling VideoLLaMA3/flash-attn when this module is
    # imported only for prompt helpers.
    from evaluation.register import INFERENCES
    from videollama3 import disable_torch_init

    parser = argparse.ArgumentParser(
        description="VideoLLaMA3 text-only LLM; save I/O to JSON. "
        "Use --descriptions-json for QA over segment description JSON; "
        "otherwise OTB-class labeling unless --instruction is set."
    )
    parser.add_argument(
        "--descriptions-json",
        type=Path,
        default=None,
        help="JSON file with segment entries containing 'description' fields "
        "(e.g. workdirs/.../segment_descriptions.json). All descriptions are sent with the query.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="HF id or local path (default: DAMO-NLP-SG/VideoLLaMA3-7B).",
    )
    parser.add_argument(
        "--max-visual-tokens",
        type=int,
        default=None,
        help="Optional cap for vision tokens (same as evaluate.py); unused for text-only but forwarded to processor.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device for HF device_map (default: "cuda:0").',
    )
    parser.add_argument(
        "--instruction",
        default="",
        help="System / task instruction. If empty, a fixed OTB-class prompt is built from --classes-file.",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=_DEFAULT_CLASSES_FILE,
        help=f"OTB class list (default: {_DEFAULT_CLASSES_FILE}). Used when --instruction is omitted.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="",
        help="User message body (the text you want the model to answer).",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="If set, user text is read from this file (overrides --input).",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        type=Path,
        default=None,
        help="Path for the JSON record. Default: llm_text_io_<utc-timestamp>.json in cwd.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--no-otb-fallback",
        action="store_true",
        help="With the default OTB instruction, do not merge object-in-label fallback classes into output.",
    )
    args = parser.parse_args()

    if args.input_file is not None:
        query_text = args.input_file.read_text(encoding="utf-8")
    else:
        query_text = args.input

    query_text = query_text.strip()
    if not query_text:
        parser.error("Provide non-empty --input / -i or --input-file.")

    segment_keys: list[str] = []
    classes: list[tuple[str, str, str]] | None = None
    if args.descriptions_json is not None:
        if not args.descriptions_json.is_file():
            parser.error(f"Descriptions JSON not found: {args.descriptions_json}")
        descriptions, segment_keys = load_descriptions_json(args.descriptions_json)
        if not descriptions:
            parser.error(f"No non-empty 'description' fields in {args.descriptions_json}")
        if args.instruction.strip():
            instruction_text = args.instruction.strip()
        else:
            instruction_text = build_segment_qa_system_instruction()
        user_text = build_segment_qa_user_content(descriptions, query_text, segment_keys)
    else:
        user_text = query_text
        if args.instruction.strip():
            instruction_text = args.instruction.strip()
        else:
            if not args.classes_file.is_file():
                parser.error(
                    f"Classes file not found: {args.classes_file} "
                    "(or pass --instruction, or use --descriptions-json for segment QA)."
                )
            classes = load_otb_classes(args.classes_file)
            if not classes:
                parser.error(f"No classes parsed from {args.classes_file}")
            instruction_text = build_otb_instruction(classes)

    out_path = args.output_json
    if out_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = Path.cwd() / f"llm_text_io_{ts}.json"

    gen_kwargs = {
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p

    model_path = args.model_path or "DAMO-NLP-SG/VideoLLaMA3-7B"
    model_init_fn, mm_infer_fn = INFERENCES(model_path)



    disable_torch_init()
    model, processor = model_init_fn(
        model_path,
        args.max_visual_tokens,
        device_map={"": args.device},
    )
    conversation = build_conversation(instruction_text or None, user_text)

    # Same as videollama3/infer.py: message list + processor applies VideoLLaMA3 chat template.
    inputs = processor(
        images=None,
        text=conversation,
        merge_size=1,
        return_tensors="pt",
    )

    output_text = mm_infer_fn(
        inputs,
        model=model,
        tokenizer=processor.tokenizer,
        modal="text",
        **gen_kwargs,
    )

    final_output = output_text
    otb_meta: dict[str, object] | None = None
    if classes is not None and not args.no_otb_fallback:
        final_output, otb_meta = merge_otb_llm_and_fallback(output_text, user_text, classes)

    record = {
        "instruction": instruction_text or None,
        "input": user_text,
        "query": query_text if args.descriptions_json is not None else None,
        "descriptions_json": str(args.descriptions_json.resolve())
        if args.descriptions_json is not None
        else None,
        "output": final_output,
        "output_llm": output_text if classes is not None else None,
        "otb_extraction": otb_meta,
        "generation": {
            "do_sample": gen_kwargs["do_sample"],
            "max_new_tokens": gen_kwargs["max_new_tokens"],
            **({k: gen_kwargs[k] for k in ("temperature", "top_p") if k in gen_kwargs}),
        },
        "model_path": model_path,
        "conversation": conversation,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.resolve()}", file=sys.stderr)
    print(final_output)


if __name__ == "__main__":
    main()
