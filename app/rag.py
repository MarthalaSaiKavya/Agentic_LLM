"""Retrieval orchestration including local documents and web search."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import requests

from .docstore import MemoryVectorStore


logger = logging.getLogger(__name__)


def search_web(query: str, k: int = 3) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Attempt Tavily search first, falling back to DuckDuckGo."""

    logger.info("Executing web search for '%s' (max_results=%d)", query, k)
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": k,
                    "include_images": False,
                    "include_answer": False,
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            results = []
            for item in payload.get("results", [])[:k]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "body": item.get("content", ""),
                        "href": item.get("url", ""),
                    }
                )
            if results:
                logger.info("Tavily returned %d result(s).", len(results))
                return results, None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily search failed: %s", exc)
            error = f"Tavily search failed: {exc}"
        else:
            error = "Tavily returned no results."
    else:
        error = "TAVILY_API_KEY not set; falling back to DuckDuckGo."

    try:
        try:
            from ddgs import DDGS  # type: ignore
            logger.debug("Using ddgs package for DuckDuckGo search.")
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore
            logger.debug("Using legacy duckduckgo_search package for web search.")

        results: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=k):
                results.append(
                    {
                        "title": result.get("title", ""),
                        "body": result.get("body", ""),
                        "href": result.get("href", ""),
                    }
                )
        if results:
            logger.info("DuckDuckGo returned %d result(s).", len(results))
            return results, None
        return [], error
    except Exception as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo search failed: %s", exc)
        fallback_error = f"DuckDuckGo search failed: {exc}"
        if error:
            fallback_error = f"{error} | {fallback_error}"
        return [], fallback_error


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str
    metadata: Dict[str, str]


@dataclass
class RetrievalResult:
    question: str
    chunks: List[RetrievedChunk]
    web_results: List[Dict[str, str]]
    used_websearch: bool
    web_error: Optional[str] = None


@dataclass
class RAGPipeline:
    vector_store: MemoryVectorStore
    base_top_k: int = 4
    max_context_length: int = 1800

    def retrieve(self, question: str, *, top_k: int | None = None, use_web: bool = False) -> RetrievalResult:
        k = top_k or self.base_top_k
        logger.info("Retrieving context for question '%s' (k=%d use_web=%s)", question[:80], k, use_web)
        doc_results = self.vector_store.similarity_search(question, k=k)
        chunks = [
            RetrievedChunk(
                text=chunk.text,
                score=score,
                source=chunk.doc_id,
                metadata={**chunk.metadata, "chunk_id": chunk.chunk_id},
            )
            for chunk, score in doc_results
        ]
        logger.info("Vector store returned %d chunk(s).", len(chunks))

        web_results: List[Dict[str, str]] = []
        web_error: Optional[str] = None
        if use_web:
            web_results, web_error = search_web(question, k=3)
        logger.info(
            "Retrieval complete (web_results=%d, web_error=%s).",
            len(web_results),
            bool(web_error),
        )
        return RetrievalResult(
            question=question,
            chunks=chunks,
            web_results=web_results,
            used_websearch=use_web,
            web_error=web_error,
        )

    def build_prompt(self, question: str, chunks: Sequence[RetrievedChunk], web_results: Sequence[Dict[str, str]]) -> str:
        logger.debug(
            "Building prompt with %d doc chunk(s) and %d web result(s).", len(chunks), len(web_results)
        )
        context_sections: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            context_sections.append(
                f"[Doc {idx}] (score={chunk.score:.2f}, source={chunk.source})\n{chunk.text}\n"
            )

        for idx, web in enumerate(web_results, start=len(context_sections) + 1):
            body = web.get("body") or ""
            context_sections.append(
                f"[Web {idx}] {web.get('title','')}\n{body}\nLink: {web.get('href','')}\n"
            )

        context_blob = "\n".join(context_sections)
        prompt = (
            "You are an uncertainty-aware assistant. Carefully read the provided references and answer the question.\n"
            "Cite sources using [Doc X] or [Web X]. If the answer is uncertain or missing, acknowledge it.\n"
            f"Question: {question}\n\n"
            f"References:\n{context_blob}\n"
            "Answer:"
        )
        return prompt[: self.max_context_length] if len(prompt) > self.max_context_length else prompt
