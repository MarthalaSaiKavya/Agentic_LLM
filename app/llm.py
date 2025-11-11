"""LLM responder abstractions for local and hosted models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import numpy as np

from src.token_self_repair.llm import LlamaProvider, load_llama

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelResponse:
    text: str
    tokens: List[str]
    logits: Optional[np.ndarray]
    metadata: dict


@lru_cache(maxsize=2)
def _load_llama_provider(model_name: str, quantize: bool) -> LlamaProvider:
    logger.info("Loading local Llama provider model=%s quantize=%s", model_name, quantize)
    return load_llama(model_name=model_name, quantize=quantize)


class LocalLlamaResponder:
    """Generate answers using the local Llama provider with logits."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", quantize: bool = False):
        self.model_name = model_name
        self.quantize = quantize
        self.provider = _load_llama_provider(model_name, quantize)
        logger.debug("Initialized LocalLlamaResponder(model=%s, quantize=%s)", model_name, quantize)

    def generate(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.1) -> ModelResponse:
        do_sample = temperature > 0.2
        logger.info(
            "Local generation start model=%s temp=%.2f max_tokens=%d sample=%s prompt_preview=%s",
            self.model_name,
            temperature,
            max_tokens,
            do_sample,
            prompt[:120],
        )
        token_ids, logits_tensor = self.provider.generate_with_logits(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        tokens = [
            self.provider.tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in token_ids
        ]
        text = self.provider.tokenizer.decode(token_ids, skip_special_tokens=False)
        logits = logits_tensor.cpu().numpy()
        logger.info("Local generation finished tokens=%d logits_shape=%s", len(tokens), logits.shape)

        return ModelResponse(
            text=text,
            tokens=tokens,
            logits=logits,
            metadata={
                "model": self.model_name,
                "quantized": self.quantize,
            },
        )


class OpenAIResponder:
    """Generate answers via OpenAI Chat Completions API with optional logprobs."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAIResponder.") from exc

        self.client = OpenAI()
        self.model_name = model_name
        logger.debug("Initialized OpenAIResponder(model=%s)", model_name)

    def generate(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.1) -> ModelResponse:
        logger.info(
            "OpenAI generation start model=%s temp=%.2f max_tokens=%d prompt_preview=%s",
            self.model_name,
            temperature,
            max_tokens,
            prompt[:120],
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )
        choice = response.choices[0]
        text = choice.message.content or ""

        tokens: List[str] = []
        probability_rows: List[List[float]] = []
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                token = getattr(token_info, "token", None)
                if token is None:
                    continue
                tokens.append(token)

                prob_map: dict[str, float] = {}
                base_logprob = getattr(token_info, "logprob", None)
                if base_logprob is not None:
                    prob_map[token] = float(np.exp(base_logprob))

                for alt in getattr(token_info, "top_logprobs", []) or []:
                    alt_token = getattr(alt, "token", None) or f"alt_{len(prob_map)}"
                    alt_logprob = getattr(alt, "logprob", None)
                    if alt_logprob is None:
                        continue
                    prob_map[alt_token] = float(np.exp(alt_logprob))

                row = list(prob_map.values())
                if len(row) < 2:
                    row.append(1e-6)
                row.sort(reverse=True)
                probability_rows.append(row)

        logits: Optional[np.ndarray] = None
        if probability_rows:
            max_len = max(len(row) for row in probability_rows)
            logits = np.full((len(probability_rows), max_len), 1e-6, dtype=np.float32)
            for idx, row in enumerate(probability_rows):
                logits[idx, : len(row)] = row
            logger.debug(
                "Constructed probability matrix for uncertainty with shape %s",
                logits.shape,
            )
        logger.info(
            "OpenAI generation finished tokens=%d logits_available=%s",
            len(tokens),
            logits is not None,
        )

        return ModelResponse(
            text=text,
            tokens=tokens,
            logits=logits,
            metadata={
                "model": self.model_name,
                "openai": True,
            },
        )
