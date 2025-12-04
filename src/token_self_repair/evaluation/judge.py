"""Utilities for judging answer correctness with an LLM (or fallback heuristics)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(slots=True)
class JudgeResult:
    correct: Optional[bool]
    explanation: str


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _heuristic_judge(question: str, prediction: str, reference: str) -> JudgeResult:
    if not reference:
        return JudgeResult(correct=None, explanation="No reference provided; unable to judge.")

    ref_norm = _normalize_text(reference)
    pred_norm = _normalize_text(prediction)

    if ref_norm and ref_norm in pred_norm:
        return JudgeResult(correct=True, explanation="Reference answer appears verbatim in the prediction.")

    # Numeric heuristic
    import re

    ref_numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", reference)
    pred_numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", prediction)
    if ref_numbers and pred_numbers and set(ref_numbers) & set(pred_numbers):
        return JudgeResult(
            correct=True,
            explanation=f"Shared numeric values between prediction and reference: {set(ref_numbers) & set(pred_numbers)}.",
        )

    return JudgeResult(correct=False, explanation="Heuristic judge did not find evidence of correctness.")


def judge_answer(
    question: str,
    prediction: str,
    reference: str,
    *,
    model: str = "gpt-4o-mini",
) -> JudgeResult:
    if not reference:
        return JudgeResult(correct=None, explanation="No reference answer supplied.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _heuristic_judge(question, prediction, reference)

    try:
        from openai import OpenAI
    except ImportError:
        return _heuristic_judge(question, prediction, reference)

    client = OpenAI()
    prompt = (
        "You are a strict grader. Determine whether the student's answer is correct "
        "given the reference answer. Respond ONLY with a JSON object of the form "
        '{"correct": true/false, "explanation": "..."}.\n\n'
        f"Question: {question}\n"
        f"Reference Answer: {reference}\n"
        f"Student Answer: {prediction}\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict grading assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content)
        correct = data.get("correct")
        explanation = data.get("explanation", "")
        if isinstance(correct, bool):
            return JudgeResult(correct=correct, explanation=explanation)
    except Exception as exc:
        return JudgeResult(correct=None, explanation=f"LLM judge failed: {exc}")

    return _heuristic_judge(question, prediction, reference)


def _heuristic_patch_judge(prediction: str, reference: str) -> JudgeResult:
    """Fallback judge for program repair: normalized string equality."""
    if not reference:
        return JudgeResult(correct=None, explanation="No reference patch supplied.")
    pred_norm = _normalize_text(prediction)
    ref_norm = _normalize_text(reference)
    if not pred_norm:
        return JudgeResult(correct=False, explanation="Predicted patch is empty.")
    if pred_norm == ref_norm:
        return JudgeResult(correct=True, explanation="Predicted patch matches reference after normalization.")
    return JudgeResult(
        correct=False,
        explanation="Predicted patch does not textually match the reference; functional correctness not checked.",
    )


def judge_patch(
    buggy_description: str,
    prediction_patch: str,
    reference_patch: str,
    *,
    model: str = "gpt-4o-mini",
) -> JudgeResult:
    """Judge whether a predicted patch is functionally equivalent to the reference.

    Uses an LLM grader when OPENAI_API_KEY and openai are available, and falls
    back to a simple normalized string comparison otherwise.
    """
    if not reference_patch:
        return JudgeResult(correct=None, explanation="No reference patch supplied.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _heuristic_patch_judge(prediction_patch, reference_patch)

    try:
        from openai import OpenAI
    except ImportError:
        return _heuristic_patch_judge(prediction_patch, reference_patch)

    client = OpenAI()
    prompt = (
        "You are a strict code repair evaluator. Compare the PREDICTED_PATCH to the "
        "REFERENCE_PATCH and decide if they are functionally equivalent and satisfy "
        "the bug description.\n\n"
        "Respond ONLY with a JSON object of the form "
        '{"correct": true/false, "explanation": "..."}.\n\n'
        f"BUG_DESCRIPTION:\n{buggy_description}\n\n"
        f"REFERENCE_PATCH:\n{reference_patch}\n\n"
        f"PREDICTED_PATCH:\n{prediction_patch}\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert software testing and repair judge."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content)
        correct = data.get("correct")
        explanation = data.get("explanation", "")
        if isinstance(correct, bool):
            return JudgeResult(correct=correct, explanation=explanation)
    except Exception as exc:
        return JudgeResult(correct=None, explanation=f"LLM patch judge failed: {exc}")

    return _heuristic_patch_judge(prediction_patch, reference_patch)


