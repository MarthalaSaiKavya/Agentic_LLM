"""Streamlit UI for uncertainty-aware RAG assistant and reasoning benchmarks."""

from __future__ import annotations

import io
import json
import logging
import warnings
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
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
from src.token_self_repair.evaluation import ReasoningEvaluationRunner, ProgramRepairEvaluationRunner
from src.token_self_repair.pipelines import default_reasoning_coordinator, default_program_repair_coordinator
from src.token_self_repair.evaluation.metrics import (
    pass_at_k,
    latency_overhead,
    user_trust_correlation,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message="Importing from 'instructor.client' is deprecated",
)

LOG_DIR = PROJECT_ROOT / "logs"
LATENCY_BASELINE_FILE = LOG_DIR / "latency_baselines.json"
TRUST_FEEDBACK_FILE = LOG_DIR / "user_trust_feedback.jsonl"


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def render_hotspot_timeline(u_map: Optional["UncertaintyMap"]) -> None:
    if not u_map or not u_map.line_scores:
        st.info("No hotspot data available for this answer.")
        return
    rows = []
    for line_no, info in sorted(u_map.line_scores.items()):
        rows.append(
            {
                "line": line_no,
                "uncertainty": info.total,
                "aleatoric": info.aleatoric,
                "epistemic": info.epistemic,
                "text": info.text,
            }
        )
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x="line:Q", y="uncertainty:Q", tooltip=["line", "uncertainty", "text"])
        .properties(height=200)
    )
    st.altair_chart(chart, theme="streamlit")
    st.dataframe(df[["line", "uncertainty", "text"]].head(15), width="stretch")


def render_token_map(u_map: Optional["UncertaintyMap"]) -> None:
    if not u_map or u_map.scores is None:
        st.info("Token map unavailable for this response.")
        return
    tokens = u_map.tokens or []
    totals = u_map.scores.total.tolist()
    aleatoric = u_map.scores.au.tolist()
    epistemic = u_map.scores.eu.tolist()
    rows = []
    for idx, token in enumerate(tokens):
        token_clean = token.replace("\n", "\\n") or "[token]"
        rows.append(
            {
                "index": idx,
                "token": token_clean,
                "total": totals[idx],
                "aleatoric": aleatoric[idx],
                "epistemic": epistemic[idx],
            }
        )
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("total", ascending=False).head(25)
    st.dataframe(df_sorted, width="stretch")


def render_message_history() -> None:
    history = st.session_state.get("chat_history", [])
    if not history:
        st.info("No conversation history yet.")
        return
    for idx, entry in enumerate(history[-10:], start=max(len(history) - 9, 1)):
        st.markdown(f"**Turn {idx} – User:** {entry.get('question','')}")
        st.markdown(f"**Assistant:** {entry.get('answer','')}")
        st.markdown("---")


def _read_json_file(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read %s: %s", path, exc)
        return {}


def _write_json_file(path: Path, payload: Dict[str, float]) -> None:
    ensure_log_dir()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def register_latency_baseline(benchmark: str, measured: float) -> float:
    baselines = _read_json_file(LATENCY_BASELINE_FILE)
    baseline = baselines.get(benchmark)
    if baseline is None:
        baseline = measured
        baselines[benchmark] = baseline
        logger.info("Recording initial latency baseline for %s: %.3fs", benchmark, baseline)
    else:
        updated = 0.9 * baseline + 0.1 * measured
        baselines[benchmark] = updated
        logger.info(
            "Updating latency baseline for %s: old=%.3f new_measure=%.3f updated=%.3f",
            benchmark,
            baseline,
            measured,
            updated,
        )
    _write_json_file(LATENCY_BASELINE_FILE, baselines)
    return float(baseline)


def load_trust_feedback(benchmark: Optional[str] = None) -> List[Dict[str, float]]:
    ensure_log_dir()
    entries: List[Dict[str, float]] = []
    if not TRUST_FEEDBACK_FILE.exists():
        return entries
    with TRUST_FEEDBACK_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if benchmark and entry.get("benchmark") != benchmark:
                    continue
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def append_trust_feedback(entries: List[Dict[str, float]]) -> None:
    if not entries:
        return
    ensure_log_dir()
    with TRUST_FEEDBACK_FILE.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    logger.info("Appended %d trust feedback entr(y/ies).", len(entries))

# ---------------------------------------------------------------------------
# Session Helpers
# ---------------------------------------------------------------------------
def get_embedder() -> TransformerEmbedder:
    if "embedder" not in st.session_state:
        logger.info("Initializing shared TransformerEmbedder for session.")
        st.session_state.embedder = TransformerEmbedder()
    else:
        logger.debug("Reusing TransformerEmbedder from session cache.")
    return st.session_state.embedder


def get_vector_store() -> MemoryVectorStore:
    if "doc_store" not in st.session_state:
        logger.info("Creating new MemoryVectorStore for session.")
        st.session_state.doc_store = MemoryVectorStore(embedder=get_embedder())
    else:
        logger.debug("Reusing cached MemoryVectorStore.")
    return st.session_state.doc_store


def get_rag_pipeline() -> RAGPipeline:
    if "rag" not in st.session_state:
        logger.info("Initializing RAGPipeline.")
        st.session_state.rag = RAGPipeline(vector_store=get_vector_store())
    else:
        logger.debug("Reusing cached RAGPipeline.")
    return st.session_state.rag


def get_local_responder(model_name: str, quantize: bool) -> LocalLlamaResponder:
    key = f"local_responder::{model_name}::{quantize}"
    if key not in st.session_state:
        logger.info("Creating LocalLlamaResponder %s (quantize=%s).", model_name, quantize)
        st.session_state[key] = LocalLlamaResponder(model_name=model_name, quantize=quantize)
    else:
        logger.debug("Reusing LocalLlamaResponder %s (quantize=%s).", model_name, quantize)
    return st.session_state[key]


def get_openai_responder(model_name: str) -> OpenAIResponder:
    key = f"openai_responder::{model_name}"
    if key not in st.session_state:
        logger.info("Creating OpenAIResponder for model %s.", model_name)
        st.session_state[key] = OpenAIResponder(model_name=model_name)
    else:
        logger.debug("Reusing OpenAIResponder %s.", model_name)
    return st.session_state[key]


def init_history() -> None:
    if "chat_history" not in st.session_state:
        logger.debug("Initializing chat history in session state.")
        st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------
def ingest_uploaded_files(files: List[io.BytesIO]) -> None:
    logger.info("Processing %d uploaded file(s) for ingestion.", len(files))
    store = get_vector_store()
    for file in files:
        name = file.name or f"upload_{len(store._chunks)}"
        text = extract_text(file)
        if text.strip():
            store.add_document(doc_id=name, text=text, metadata={"source": name})
            logger.info("Ingested document '%s' into vector store.", name)
        else:
            logger.warning("Skipping empty upload '%s'.", name)


def extract_text(file: io.BytesIO) -> str:
    name = (file.name or "").lower()
    data = file.read()
    file.seek(0)
    if name.endswith(".pdf"):
        try:
            import PyPDF2  # type: ignore
        except ImportError:
            logger.warning("PyPDF2 missing; cannot parse PDF '%s'.", file.name)
            return ""
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            logger.debug("Extracted %d characters from PDF '%s'.", len(text), file.name)
            return text
        except Exception:
            logger.exception("Failed to extract text from PDF '%s'.", file.name)
            return ""
    try:
        text = data.decode("utf-8")
        logger.debug("Extracted %d characters from text file '%s'.", len(text), file.name)
        return text
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="ignore")
        logger.debug("Decoded %d characters from binary file '%s' using latin-1.", len(text), file.name)
        return text


# ---------------------------------------------------------------------------
# Retrieval + Generation
# ---------------------------------------------------------------------------
def run_retrieval(question: str, top_k: int, include_web: bool) -> Dict:
    pipeline = get_rag_pipeline()
    logger.info("Running retrieval pipeline (top_k=%d, include_web=%s).", top_k, include_web)
    retrieval = pipeline.retrieve(question, top_k=top_k, use_web=include_web)
    prompt = pipeline.build_prompt(question, retrieval.chunks, retrieval.web_results)
    logger.debug("Prompt length after retrieval: %d characters.", len(prompt))
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
        logger.info("Dispatching generation to local model %s.", local_model_name)
        return responder.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    logger.info("Dispatching generation to OpenAI model %s.", model_choice)
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
    else:
        logger.debug("Skipping uncertainty computation (tokens=%d, logits=%s).", len(tokens), logits is not None)

    low_confidence = False
    if uncertainty and uncertainty.avg_logtoku > confidence_threshold:
        low_confidence = True
    if ragas_metrics:
        faithfulness = ragas_metrics.get("faithfulness", 1.0)
        if faithfulness < 0.5:
            low_confidence = True
    logger.info(
        "Assembled response metrics: low_confidence=%s ragas_available=%s",
        low_confidence,
        bool(ragas_metrics),
    )

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
    logger.debug("Displaying response for question '%s'.", entry.get("question", "")[:80])
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
        logger.warning("Low confidence flagged for question '%s'.", entry["question"][:80])

    u_map = entry.get("uncertainty_map")
    with st.expander("Hotspot Timeline", expanded=False):
        render_hotspot_timeline(u_map)
    with st.expander("Token Map", expanded=False):
        render_token_map(u_map)


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    logger.info("Starting Streamlit app.")
    st.set_page_config(page_title="Uncertainty RAG Assistant", layout="wide")
    init_history()

    tab_chat, tab_eval, tab_repair = st.tabs(["Assistant", "Reasoning Benchmarks", "Program Repair"])

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
            logger.info("Manual note added to knowledge base as %s.", doc_id)

    with tab_chat:
        st.title("Uncertainty-Aware RAG Assistant")
        st.caption("Ask questions, inspect uncertainties, and trigger repair flows.")

        for entry in st.session_state.chat_history:
            display_response(entry)

        with st.expander("Message History Pane", expanded=False):
            render_message_history()

        user_question = st.chat_input("Ask a question or provide instructions")
        if user_question:
            logger.info("Received user question (%d chars).", len(user_question))
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
                        logger.info("User triggered repair flow for question '%s'.", user_question[:80])
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
                        logger.info("Repair flow completed and appended to history.")

    with tab_eval:
        st.title("Reasoning Benchmark Dashboard")
        st.caption("Evaluate models on reasoning datasets with uncertainty metrics.")

        eval_model_name = st.selectbox(
            "Evaluation Model (local only for uncertainty metrics)",
            ["meta-llama/Llama-3.2-3B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
            index=0,
        )
        max_samples = st.slider("Number of samples", 1, 20, 5)
        dataset_options = ["gsm8k", "humaneval", "truthfulqa", "bioasq", "repair"]
        dataset = st.selectbox("Dataset", dataset_options)

        if st.button("Run Benchmark", key="run_benchmark"):
            logger.info("Starting reasoning benchmark dataset=%s model=%s samples=%d", dataset, eval_model_name, max_samples)
            coordinator = default_reasoning_coordinator(get_local_responder(eval_model_name, quantize=False).provider)

            def factory():
                return coordinator

            runner = ReasoningEvaluationRunner(coordinator_factory=factory)
            result = runner.run(dataset, max_samples=max_samples)
            st.metric("Accuracy", f"{result.accuracy:.2f}")
            st.metric("AUROC", f"{result.auroc:.2f}")
            st.metric("Average Uncertainty", f"{result.average_uncertainty:.3f}")
            st.metric("Calibration Error", f"{result.calibration_error:.3f}")
            st.metric("Average Latency (s)", f"{result.average_latency:.2f}")

            pass_outcomes = [[sample.correct] for sample in result.samples]
            pass_at_one = pass_at_k(pass_outcomes, k=1) if pass_outcomes else 0.0
            st.metric("Pass@1", f"{pass_at_one:.2f}")

            augmented_latencies = [sample.latency_seconds for sample in result.samples]
            baseline_value = register_latency_baseline(result.benchmark, result.average_latency)
            latency_penalty = (
                latency_overhead([baseline_value], [result.average_latency]) if result.average_latency else 0.0
            )
            st.metric("Latency Overhead", f"{latency_penalty:.2f}")

            confidences = [sample.confidence for sample in result.samples]
            recorded_feedback = load_trust_feedback(result.benchmark)
            if recorded_feedback:
                trust_confidences = [entry["confidence"] for entry in recorded_feedback]
                trust_scores = [entry["trust"] for entry in recorded_feedback]
            else:
                trust_confidences = confidences
                trust_scores = [4.5 if sample.correct else 2.0 for sample in result.samples]
            trust_corr = (
                user_trust_correlation(trust_confidences, trust_scores) if trust_confidences else 0.0
            )
            st.metric("Trust Correlation", f"{trust_corr:.2f}")

            logger.info(
                "Benchmark metrics computed: pass@1=%.3f latency_overhead=%.3f trust_corr=%.3f baseline=%.3f",
                pass_at_one,
                latency_penalty,
                trust_corr,
                baseline_value,
            )

            if result.calibration_bins:
                bin_conf, bin_acc = zip(*result.calibration_bins)
                calibration_df = pd.DataFrame(
                    {"Confidence": bin_conf, "Empirical Accuracy": bin_acc}
                )
                st.line_chart(calibration_df.set_index("Confidence"))

            with st.expander("Record User Trust Feedback"):
                if not result.samples:
                    st.info("Run a benchmark to enable feedback.")
                else:
                    sample_options = {
                        f"Sample {idx + 1}: {sample.prompt[:60]}": sample
                        for idx, sample in enumerate(result.samples)
                    }
                    selected_label = st.selectbox("Select sample", list(sample_options.keys()))
                    trust_rating = st.slider("Trust rating (1=low, 5=high)", 1.0, 5.0, 4.0, 0.5)
                    note = st.text_input("Optional note")
                    if st.button("Submit Trust Feedback", key="trust_submit"):
                        chosen = sample_options[selected_label]
                        entry = {
                            "benchmark": result.benchmark,
                            "prompt": chosen.prompt,
                            "confidence": chosen.confidence,
                            "trust": trust_rating,
                            "note": note,
                            "timestamp": time.time(),
                        }
                        append_trust_feedback([entry])
                        st.success("Trust feedback recorded.")
                        logger.info(
                            "Recorded trust feedback for %s (confidence=%.3f trust=%.2f).",
                            result.benchmark,
                            chosen.confidence,
                            trust_rating,
                        )

            sample_rows = []
            for sample in result.samples:
                sample_rows.append(
                    {
                        "prompt": sample.prompt,
                        "prediction": sample.prediction,
                        "reference": sample.reference,
                        "correct": sample.correct,
                        "uncertainty": sample.final_uncertainty,
                        "confidence": sample.confidence,
                        "judge_explanation": sample.judge_explanation,
                        "hotspots": json.dumps(sample.hotspots),
                        "latency_seconds": sample.latency_seconds,
                    }
                )
            if sample_rows:
                st.dataframe(sample_rows, width="stretch")
                logger.info("Displayed %d benchmark sample rows.", len(sample_rows))

    with tab_repair:
        st.title("Program Repair Benchmarks")
        st.caption("Evaluate code repair strategies on curated bug datasets.")
        repair_model_name = st.selectbox(
            "Repair Model (local Llama only)",
            ["meta-llama/Llama-3.2-3B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
            index=1,
        )
        repair_samples = st.slider("Number of samples", 1, 10, 3)
        repair_dataset = st.selectbox("Repair Dataset", ["repair", "defects4j", "gitbugs"])

        if st.button("Run Program Repair Benchmark", key="run_program_repair"):
            logger.info(
                "Starting program repair benchmark dataset=%s model=%s samples=%d",
                repair_dataset,
                repair_model_name,
                repair_samples,
            )

            responder = get_local_responder(repair_model_name, quantize=False)

            def repair_factory():
                return default_program_repair_coordinator(responder.provider)

            repair_runner = ProgramRepairEvaluationRunner(coordinator_factory=repair_factory)
            repair_result = repair_runner.run(repair_dataset, max_samples=repair_samples)
            st.metric("Pass Rate", f"{repair_result.pass_rate:.2f}")
            st.metric("Pass@1", f"{repair_result.pass_at_one:.2f}")
            st.metric("Average Latency (s)", f"{repair_result.average_latency:.2f}")

            sample_rows = []
            for sample in repair_result.samples:
                sample_rows.append(
                    {
                        "prompt": sample.prompt,
                        "buggy_code": sample.buggy_code,
                        "failing_tests": "\n".join(sample.failing_tests),
                        "patch": sample.patch,
                        "reference_patch": sample.reference_patch,
                        "success": sample.success,
                        "latency_seconds": sample.latency_seconds,
                    }
                )
            if sample_rows:
                st.dataframe(sample_rows, width="stretch")
            st.info(
                "Repair logs are available in the terminal output for detailed troubleshooting of each attempt."
            )

if __name__ == "__main__":
    main()
