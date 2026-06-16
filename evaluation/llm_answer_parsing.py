from __future__ import annotations

import re
from typing import Any


def _ollama_chat(model: str, prompt: str) -> str | None:
    """Run a single user-message chat against a local Ollama model; return raw text or None."""
    try:
        import ollama
    except ModuleNotFoundError:
        return None
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        return None
    return response["message"]["content"]


def build_prompt(question, options, prediction):
    prompt = f'''
You are an AI agent designed to evaluate responses to multiple-choice questions. Your task is to determine the best answer to the following question based on the provided options. You should not explain your answer or provide any additional information, you should only provide the letter of the multiple choice answer that is most similar to the free-form response given to you, start your reply with "The best answer is". If the free-form response is different from all of the provided answers, the letter of the answer should be 'Z'. Here are some examples of your task:

Example 1:
Question: What is the main object in image?
Options: (A) teddy bear (B) rabbit (C) cat (D) dog
Answer: the main object in the image is a cute teddy bear
The best answer is: (A)

Example 2:
Question: What action is being performed by the person in the video?
Options: (A) walking (B) cleaning up (C) drinking from bottle (D) drinking from cup
Answer: the person in the video is seen in the kitchen, holding a cup and drinking from it
The best answer is: (C)

Example 3:
Question: What action will the person perform next?
Options: (A) sit down (B) stand up (C) walk to the door (D) walk to the window
Answer: the person is in the kitchen and holding a cup, they will most likely walk to the fridge to get some milk
The best answer is: (Z)

Example 4:
Question: Given that the person in the video "walked to the kitchen" and then "picked up the dirty plate", what action are they most likely to perform next?
Options: (A) leave the kitchen (B) walk to the sink (C) start cooking eggs (D) turn on the stove
Answer: seeing as how the person walked to the kitchen and then picked up the dirty plate, the most likely action they will perform next is to walk to the sink to wash the plate. This is because there is a sink behind them and their intention seems to be to clean the plate.
The best answer is: (B)
-----------------------------------------

Your task is to evaluate the following question and provide the best answer based on the options provided. You should not provide any additional information or explanation, only the letter of the multiple choice answer that is most similar to the free-form response given to you.
Question: {question}
Options: {options}
Answer: {prediction}
Please complete the following: The best answer is:
    '''
    return prompt


def normalize_freeform_answer_text(s) -> str:
    """Strip leading boilerplate from free-form model answers (shared with MCQ parsing)."""
    if isinstance(s, dict):
        s = ""
    s = str(s).strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    return s.strip()


def extract_yes_no_regex(answer: str) -> str | None:
    """
    First ``yes``/``no`` token after normalization (start of string, else first whole-word).

    ``not`` does not match ``no``. Aligns with human judgment on plain yes/no replies.
    """
    text = normalize_freeform_answer_text(answer)
    if not text:
        return None
    m = re.match(r"^\W*(yes|no)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def extract_yes_no_with_ollama(
    answer: str,
    model: str = "llama3.1",
    *,
    use_normalize: bool = True,
) -> str | None:
    """
    Use a local Ollama model to map a free-form reply to ``yes`` or ``no`` only.

    Returns ``None`` if the model output cannot be parsed as yes/no.

    When ``use_normalize`` is False, only leading/trailing whitespace is stripped from
    the input (no prefix stripping); use that for LLM-only evaluation pipelines.
    """
    text = normalize_freeform_answer_text(answer) if use_normalize else str(answer).strip()
    if not text:
        return None

    prompt = (
        "The text below is a model's answer to a yes/no question.\n"
        "Infer the intended binary answer and reply with exactly one word: yes or no "
        "(lowercase, no punctuation, no other words).\n"
        "If the text is too ambiguous to choose, reply with: unknown\n\n"
        f"Text:\n{text}"
    )
    raw = _ollama_chat(model, prompt)
    if raw is None:
        return None
    low = raw.strip().lower()
    m = re.search(r"\b(yes|no|unknown)\b", low)
    if not m:
        return None
    w = m.group(1)
    if w == "unknown":
        return None
    return w


def extract_concise_answer_with_ollama(
    question: str,
    description_answer: str,
    *,
    model: str = "llama3.1",
) -> str | None:
    """
    Extract a concise canonical answer phrase from a free-form description.

    Returns ``None`` if extraction is unknown / unparseable.
    """
    q = str(question).strip()
    text = str(description_answer).strip()
    if not text:
        return None
    prompt = (
        "You are extracting a concise answer to a question from a model's long response.\n"
        "Return exactly one short phrase (1-6 words), lowercase, no punctuation.\n"
        "If the response does not provide a clear answer, return exactly: unknown\n\n"
        f"Question:\n{q}\n\n"
        f"Model response:\n{text}\n\n"
        "Output only the concise answer phrase or unknown."
    )
    raw = _ollama_chat(model, prompt)
    if raw is None:
        return None
    ans = raw.strip().lower()
    ans = re.sub(r"^[\"'`]+|[\"'`]+$", "", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    if not ans or ans == "unknown":
        return None
    return ans


def semantic_label_match_ollama(
    predicted_label: str,
    ground_truth: str,
    *,
    model: str = "llama3.1",
) -> bool | None:
    """
    Judge whether ``predicted_label`` and ``ground_truth`` are semantically equivalent.

    Returns ``True`` for match, ``False`` for mismatch, and ``None`` for unknown.
    """
    pred = str(predicted_label).strip()
    gt = str(ground_truth).strip()
    if not pred or not gt:
        return None
    prompt = (
        "Compare two short labels for answer equivalence.\n"
        "Return match only if they refer to the same thing/action/attribute in this context.\n"
        "Allow close synonyms/paraphrases (e.g., doorway ~= door entrance).\n"
        "Do not match related-but-different concepts (e.g., phone != bag).\n"
        "Do not guess missing information.\n\n"
        f"Predicted label:\n{pred}\n\n"
        f"Reference label:\n{gt}\n\n"
        "Reply with exactly one word: match, mismatch, or unknown"
    )
    raw = _ollama_chat(model, prompt)
    if raw is None:
        return None
    low = raw.strip().lower()
    m = re.search(r"\b(match|mismatch|unknown)\b", low)
    if not m:
        return None
    w = m.group(1)
    if w == "unknown":
        return None
    return w == "match"


def ground_truth_supported_by_description_ollama(
    question: str,
    description_answer: Any,
    ground_truth: str,
    *,
    model: str = "llama3.1",
) -> bool | None:
    """
    Use Ollama to judge whether a verbose, description-based model answer supports the
    reference ``ground_truth`` label for the question.

    Semantic equivalence is allowed, but only when the predicted answer clearly supports
    the reference label. Returns ``True`` for match, ``False`` for mismatch, and
    ``None`` for unknown/unparseable outputs.
    """
    gt = str(ground_truth).strip()
    text = str(description_answer).strip()
    if not gt or not text:
        return None
    q = str(question).strip()
    gt_low = gt.lower()

    # Deterministic guard for binary labels to avoid false positives.
    if gt_low in ("yes", "no"):
        pred_yes_no = extract_yes_no_regex_then_ollama(text, model=model)
        if pred_yes_no is not None:
            return pred_yes_no == gt_low

    concise_pred = extract_concise_answer_with_ollama(q, text, model=model)
    if concise_pred is None:
        return None
    verdict = semantic_label_match_ollama(concise_pred, gt, model=model)
    if verdict is not None:
        return verdict

    # Fallback direct judge if concise extraction/equivalence is uncertain.
    prompt = (
        "You compare a model's written answer to a video understanding question against "
        "a reference label (the correct answer for scoring).\n\n"
        f"Question:\n{q}\n\n"
        f"Reference label (correct answer):\n{gt}\n\n"
        f"Model's answer (often long; based on scene descriptions):\n{text}\n\n"
        "Decide whether this answer supports the reference label.\n"
        "- Match only when the answer clearly implies the same final label.\n"
        "- Synonyms/paraphrases are allowed only when unambiguous.\n"
        "- Related but different objects/actions are mismatch.\n"
        "- Do not ignore any words in the answer.\n"
        "- If uncertain, reply unknown.\n\n"
        "Reply with exactly one word: match, mismatch, or unknown"
    )
    raw = _ollama_chat(model, prompt)
    if raw is None:
        return None
    low = raw.strip().lower()
    m = re.search(r"\b(match|mismatch|unknown)\b", low)
    if not m:
        return None
    w = m.group(1)
    if w == "unknown":
        return None
    return w == "match"


def extract_yes_no_regex_then_ollama(answer: str, model: str = "llama3.1") -> str | None:
    """Prefer deterministic ``yes``/``no`` token extraction; use Ollama only when regex finds nothing."""
    y = extract_yes_no_regex(answer)
    if y is not None:
        return y
    return extract_yes_no_with_ollama(answer, model=model)


def extract_characters_regex(s, choices=['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(Z)']):
    s = normalize_freeform_answer_text(s)

    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ''
    return matches[0]


def parse_with_llama(prompt, model: str = "llama3.1") -> str | None:
    """Send ``prompt`` to a local Ollama model; return raw message text."""
    return _ollama_chat(model, prompt)

def parse_with_chatgpt(prompt, model='gpt-3.5-turbo', api_key=None):
    import openai

    assert api_key is not None, 'Please provide an OpenAI API key.'
    openai.api_key = api_key

    prompt_to_llm = [
        {
            'role': 'user',
            'content': prompt
        }
    ]

    response = openai.ChatCompletion.create(model=model, messages=prompt_to_llm)
    response = response['choices'][0]['message']['content']

    answer_letter = extract_characters_regex(response)

    return answer_letter, response