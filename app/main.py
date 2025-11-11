"""Streamlit UI for uncertainty-aware RAG assistant and reasoning benchmarks."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

from app.embedding import TransformerEmbedder
from app.docstore import MemoryVectorStore
from app.rag import RAGPipeline, RetrievedChunk
from app.llm import LocalLlamaResponder, OpenAIResponder, ModelResponse
from app.metrics import compute_ragas, compute_uncertainty, UncertaintyReport
from src.token_self_repair.evaluation import ReasoningEvaluationRunner
from src.token_self_repair.pipelines import default_reasoning_coordinator

# ---------------------------------------------------------------------------
# Session Helpers
# ---------------------------------------------------------------------------
def get_embedder() -> TransformerEmbedder:
    if "embedder" not in st.session_state:
        st.session_state.embedder = TransformerEmbedder()
    return st.session_state.embedder


def get_vector_store() -> MemoryVectorStore:
    if "doc_store" not in st.session_state:
        st.session_state.doc_store = MemoryVectorStore(embedder=get_embedder())
    return st.session_state.doc_store


def get_rag_pipeline() -> RAGPipeline:
    if "rag" not in st.session_state:
        st.session_state.rag = RAGPipeline(vector_store=get_vector_store())
    return st.session_state.rag


def get_local_responder(model_name: str, quantize: bool) -> LocalLlamaResponder:
    key = f"local_responder::{model_name}::{quantize}"
    if key not in st.session_state:
        st.session_state[key] = LocalLlamaResponder(model_name=model_name, quantize=quantize)
    return st.session_state[key]


def get_openai_responder(model_name: str) -> OpenAIResponder:
    key = f"openai_responder::{model_name}"
    if key not in st.session_state:
        st.session_state[key] = OpenAIResponder(model_name=model_name)
    return st.session_state[key]


def init_history() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------
def ingest_uploaded_files(files: List[io.BytesIO]) -> None:
    store = get_vector_store()
    for file in files:
        name = file.name or f"upload_{len(store._chunks)}"
        text = extract_text(file)
        if text.strip():
            store.add_document(doc_id=name, text=text, metadata={"source": name})


def extract_text(file: io.BytesIO) -> str:
    name = (file.name or "").lower()
    data = file.read()
    file.seek(0)
    if name.endswith(".pdf"):
        try:
            import PyPDF2  # type: ignore
        except ImportError:
            return ""
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


# ---------------------------------------------------------------------------
# Retrieval + Generation
# ---------------------------------------------------------------------------
def run_retrieval(question: str, top_k: int, include_web: bool) -> Dict:
    pipeline = get_rag_pipeline()
    retrieval = pipeline.retrieve(question, top_k=top_k, use_web=include_web)
    prompt = pipeline.build_prompt(question, retrieval.chunks, retrieval.web_results)
    return {
        "retrieval": retrieval,
        "prompt": prompt,
    }


def run_generation(
    prompt: str,
    *,
    model_choice: str,
    local_model_name: str,
    quantize: bool,
    temperature: float,
    max_tokens: int,
) -> ModelResponse:
    if model_choice == "Local Llama":
        responder = get_local_responder(local_model_name, quantize)
        return responder.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    responder = get_openai_responder(model_choice)
    return responder.generate(prompt, max_tokens=max_tokens, temperature=temperature)


def assemble_response(
    question: str,
    model_response: ModelResponse,
    retrieval_info: Dict,
    confidence_threshold: float,
) -> Dict:
    text = model_response.text.replace("<|eot_id|>", "").strip()
    tokens = model_response.tokens
    logits = model_response.logits

    contexts = [chunk.text for chunk in retrieval_info["retrieval"].chunks]
    ragas_metrics = compute_ragas(question, text, contexts)

    uncertainty: Optional[UncertaintyReport] = None
    if logits is not None and len(tokens) == logits.shape[0]:
        uncertainty = compute_uncertainty(tokens, logits, text)

    low_confidence = False
    if uncertainty and uncertainty.avg_logtoku > confidence_threshold:
        low_confidence = True
    if ragas_metrics:
        faithfulness = ragas_metrics.get("faithfulness", 1.0)
        if faithfulness < 0.5:
            low_confidence = True

    return {
        "question": question,
        "answer": text,
        "model": model_response.metadata.get("model"),
        "metrics": {
            "ragas": ragas_metrics,
            "uncertainty": {
                "avg_eu": getattr(uncertainty, "avg_eu", None),
                "avg_au": getattr(uncertainty, "avg_au", None),
                "avg_logtoku": getattr(uncertainty, "avg_logtoku", None),
                "avg_entropy": getattr(uncertainty, "avg_entropy", None),
            }
            if uncertainty
            else None,
        },
        "uncertainty_map": uncertainty.map if uncertainty else None,
        "retrieval": retrieval_info["retrieval"],
        "prompt": retrieval_info["prompt"],
        "low_confidence": low_confidence,
    }


def display_response(entry: Dict) -> None:
    st.chat_message("user").write(entry["question"])
    st.chat_message("assistant").markdown(entry["answer"])

    with st.expander("Retrieval Context"):
        st.write("**Top Documents:**")
        for idx, chunk in enumerate(entry["retrieval"].chunks, start=1):
            st.markdown(f"- **Doc {idx}** (score {chunk.score:.2f}, source: {chunk.source})")
            st.caption(chunk.text[:500])
        if entry["retrieval"].used_websearch and entry["retrieval"].web_results:
            st.write("**Web Search Results:**")
            for web in entry["retrieval"].web_results:
                st.markdown(f"- [{web.get('title','(no title)')}]({web.get('href','')})")
                st.caption(web.get("body", "")[:300])
        web_error = getattr(entry["retrieval"], "web_error", None)
        if entry["retrieval"].used_websearch and web_error:
            st.warning(web_error)

    with st.expander("Metrics & Confidence"):
        metrics = entry["metrics"]
        if metrics["uncertainty"]:
            cols = st.columns(4)
            cols[0].metric("Avg EU", f"{metrics['uncertainty']['avg_eu']:.3f}")
            cols[1].metric("Avg AU", f"{metrics['uncertainty']['avg_au']:.3f}")
            cols[2].metric("Avg LogTokU", f"{metrics['uncertainty']['avg_logtoku']:.3f}")
            cols[3].metric("Avg Entropy", f"{metrics['uncertainty']['avg_entropy']:.3f}")
        else:
            st.info("Uncertainty metrics not available for this model.")

        if metrics["ragas"]:
            st.write("**RAGAS Metrics**")
            st.json(metrics["ragas"])

    if entry["low_confidence"]:
        st.warning(
            "Low confidence detected. Consider extending retrieval, revising the prompt, or switching models."
        )


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Uncertainty RAG Assistant", layout="wide")
    init_history()

    tab_chat, tab_eval = st.tabs(["Assistant", "Benchmarks"])

    with st.sidebar:
        st.header("Configuration")
        model_mode = st.radio("Assistant Model", ["Local Llama", "gpt-4o-mini"], index=0)
        local_model_name = st.selectbox(
            "Local Model",
            ["meta-llama/Llama-3.2-3B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
            index=0,
        )
        quantize = st.checkbox("Use 4-bit quantization", value=False)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_k = st.slider("Top K Documents", 1, 10, 4)
        include_web = st.checkbox("Include Web Search", value=True)
        confidence_threshold = st.slider("Low Confidence Threshold (LogTokU)", 0.01, 0.5, 0.15, 0.01)

        st.markdown("---")
        st.subheader("Knowledge Base")
        uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)
        if uploaded_files:
            ingest_uploaded_files(uploaded_files)
            st.success(f"Loaded {len(uploaded_files)} document(s).")

        manual_text = st.text_area("Add notes / paste text", height=150)
        if st.button("Add to knowledge base") and manual_text.strip():
            store = get_vector_store()
            doc_id = f"note_{len(store._chunks)}"
            store.add_document(doc_id=doc_id, text=manual_text, metadata={"source": "manual"})
            st.success("Added note to knowledge base.")

    with tab_chat:
        st.title("Uncertainty-Aware RAG Assistant")
        st.caption("Ask questions, inspect uncertainties, and trigger repair flows.")

        for entry in st.session_state.chat_history:
            display_response(entry)

        user_question = st.chat_input("Ask a question or provide instructions")
        if user_question:
            retrieval_info = run_retrieval(user_question, top_k=top_k, include_web=include_web)
            model_response = run_generation(
                retrieval_info["prompt"],
                model_choice=model_mode,
                local_model_name=local_model_name,
                quantize=quantize,
                temperature=temperature,
                max_tokens=300,
            )
            chat_entry = assemble_response(
                user_question,
                model_response,
                retrieval_info,
                confidence_threshold=confidence_threshold,
            )
            st.session_state.chat_history.append(chat_entry)
            display_response(chat_entry)

            if chat_entry["low_confidence"]:
                repair_key = f"repair::{len(st.session_state.chat_history)}"
                with st.expander("Repair Suggestions", expanded=True):
                    st.write(
                        "- Expand retrieval pool and re-run with higher top-k.\n"
                        "- Force web search for updated information.\n"
                        "- Switch to exploration strategy for broader reasoning."
                    )
                    if st.button("Run Repair", key=repair_key):
                        repair_retrieval = run_retrieval(user_question, top_k=top_k * 2, include_web=True)
                        repair_response = run_generation(
                            repair_retrieval["prompt"],
                            model_choice=model_mode,
                            local_model_name=local_model_name,
                            quantize=quantize,
                            temperature=max(temperature, 0.3),
                            max_tokens=350,
                        )
                        repair_entry = assemble_response(
                            user_question,
                            repair_response,
                            repair_retrieval,
                            confidence_threshold=confidence_threshold,
                        )
                        repair_entry["repair_of"] = len(st.session_state.chat_history)
                        st.session_state.chat_history.append(repair_entry)
                        st.success("Repair run completed.")
                        display_response(repair_entry)

    with tab_eval:
        st.title("Reasoning Benchmark Dashboard")
        st.caption("Evaluate models on reasoning datasets with uncertainty metrics.")

        eval_model_name = st.selectbox(
            "Evaluation Model (local only for uncertainty metrics)",
            ["meta-llama/Llama-3.2-3B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
            index=0,
        )
        max_samples = st.slider("Number of samples", 1, 20, 5)
        dataset = st.selectbox("Dataset", ["gsm8k", "humaneval"])

        if st.button("Run Benchmark", key="run_benchmark"):
            coordinator = default_reasoning_coordinator(get_local_responder(eval_model_name, quantize=False).provider)

            def factory():
                return coordinator

            runner = ReasoningEvaluationRunner(coordinator_factory=factory)
            result = runner.run(dataset, max_samples=max_samples)
            st.metric("Accuracy", f"{result.accuracy:.2f}")
            st.metric("Average Uncertainty", f"{result.average_uncertainty:.3f}")
            st.metric("Calibration Error", f"{result.calibration_error:.3f}")

            sample_rows = []
            for sample in result.samples:
                sample_rows.append(
                    {
                        "prompt": sample.prompt,
                        "prediction": sample.prediction,
                        "reference": sample.reference,
                        "correct": sample.correct,
                        "uncertainty": sample.final_uncertainty,
                        "judge_explanation": sample.judge_explanation,
                        "hotspots": json.dumps(sample.hotspots),
                    }
                )
            if sample_rows:
                st.dataframe(sample_rows, use_container_width=True)


if __name__ == "__main__":
    main()


