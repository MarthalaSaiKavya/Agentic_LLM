# End-to-End System Explanation

This document provides a high-level tour of the Token-Level Uncertainty Driven
Self-Repair system, from ingestion and generation to visualization and
evaluation. Use it as a reference when explaining how the pieces fit together or
when onboarding new collaborators.

---

## 1. Architectural Stack

| Layer | Key Files | Purpose |
| --- | --- | --- |
| LLM Providers | `src/token_self_repair/llm/` | Stream tokens + logits from local Llama (`LlamaProvider`), OpenAI (`OpenAIProvider`), or deterministic mocks. |
| Uncertainty Engine | `src/token_self_repair/uncertainty/` | Implements LogTokU scoring (`LogTokUEstimator`) and hierarchical aggregation (`UncertaintyAggregator`). |
| Repair Strategies | `src/token_self_repair/repair/` | Line/method/test-level strategies plus constitutional rules; decide how to rewrite when uncertainty is high. |
| Pipeline Orchestration | `src/token_self_repair/pipelines/` | `UncertaintyAwarePipeline` (sequential loop), LangGraph graph (`TokenSelfRepairGraph`), and adapters for reasoning/repair flows. |
| Messaging & Telemetry | `src/token_self_repair/messaging/status.py` | Emits “High confidence”, “Low confidence – running corrective pass”, etc., so downstream UIs know what happened. |
| Evaluation Harness | `src/token_self_repair/evaluation/` | Dataset registry, judges, metrics, and benchmark runners for reasoning and repair tasks. |
| Frontend | `app/main.py` + helpers | Streamlit dashboard for chat, reasoning benchmarks, program repair metrics, uncertainty visualizations. |

---

## 2. Data Flow (Single Response)

1. **Prompt Construction**
   - RAG pipeline (`app/rag.py`) retrieves documents from the in-memory vector store and optional web search.
   - Context is deduplicated and assembled into a single prompt.

2. **Generation + Streaming**
   - Selected LLM provider yields `TokenLogit(token, logits)` entries.
   - `UncertaintyAwarePipeline` buffers the stream so it can run LogTokU after every token.

3. **Uncertainty Scoring**
   - `LogTokUEstimator.score()` converts logits to per-token probability, entropy, aleatoric (AU), epistemic (EU), and total uncertainty.
   - `UncertaintyAggregator` can map tokens → lines → methods to find hotspots.

4. **Repair Decision**
   - If the final token’s `total_uncertainty` exceeds `thresholds.repair_activation_uncertainty`, the pipeline chooses a strategy (line/method/test) based on avg EU vs AU.
   - A repair instruction is injected and the loop restarts; max attempts governed by `ProjectConfig.max_self_repairs`.

5. **Result Packaging**
   - GenerationStep (tokens, uncertainty scores, repair notes) plus messenger history are returned to the caller (Streamlit UI, CLI example, or evaluation harness).

---

## 3. Streamlit UI Experience (`app/main.py`)

### Assistant Tab
- Chat input triggers retrieval + generation.
- Response area displays:
  - Colored tokens (green→red) based on normalized LogTokU.
  - Explicit confidence badge (`LOW`/`HIGH`) with numeric threshold comparison.
  - Retrieval context, RAGAS metrics, hotspot timeline, and token table.
- If `low_confidence=True`, a repair expander offers suggestions and re-runs retrieval with broader settings.

### Reasoning Benchmarks Tab
- Choose dataset, model, sample count.
- Runner executes `ReasoningEvaluationRunner`, logging accuracy, AUROC, calibration error, latency, pass@1, latency overhead, and trust correlation.
- Displays calibration curves, sample-by-sample tables, and optionally collects user trust ratings.

### Program Repair Tab
- Hooks into `ProgramRepairEvaluationRunner` (currently string-match baseline).
- Shows pass rate, pass@1, average latency, and per-sample table with prompts, patches, reference patches, and messenger logs.

---

## 4. Evaluation Loop

1. **Dataset Registry**
   - `evaluation/datasets.py` defines mini versions of GSM8K, HumanEval, TruthfulQA, BioASQ, and repair benchmarks (Repair mini, Defects4J mini, GitBugs mini).

2. **Reasoning Runner**
   - For each sample: instantiate coordinator → `solve()` → collect outputs.
   - Metrics: exact match accuracy, AUROC (uncertainty vs. correctness), expected calibration error, average uncertainty, average latency, pass@k, latency overhead, and trust correlation.

3. **Repair Runner**
   - For each sample: run `ProgramRepairCoordinator.repair()` → check patch equality → log latency.
   - Outputs pass rate, pass@1, average latency, and per-sample metadata.
   - (Functional correctness harness still TODO; current metric is exact text match.)

4. **Visualization**
   - Streamlit table aggregates metrics per dataset so you can see qualitative trends even with small sample sizes.

---

## 5. Example End-to-End Run

Refer to `docs/end_to_end_pipeline_example.md` and `docs/self_repair_run_example.md` for concrete numbers. Highlights:

- **Reasoning Prompt (“Explain quicksort on [8,3,5]”)**  
  - Initial pass flagged high uncertainty (`total ≈ 0.58`) on tokens describing the pivot.
  - Line-level strategy issued a rewrite instruction.
  - Second pass dropped avg LogTokU below threshold and yielded a clean explanation.

- **Factoid Prompt (“Conclusion of this project”)**  
  - Retrieval pulled two doc chunks, avg LogTokU stayed low (~0.03), so no repair triggered.
  - Token map + timeline still rendered so reviewers can verify confidence zones.

---

## 6. Limitations & Next Steps

- **Repair evaluation** currently checks literal string equality; integrating real build/test runs (Defects4J, GitBugs) remains future work.
- **Sample size in reasoning benchmarks** defaults to very small subsets (N=5). For statistically meaningful numbers, increase `max_samples`.
- **LLM quality** in examples uses TinyLlama-1.1B. Switching to stronger models (Llama‑3 8B, GPT‑4o) improves both accuracy and repair quality.
- **Additional fact checks** (e.g., letter-count validation) were prototyped but disabled by default; you can reintroduce them in `app/main.py` if needed.

---

## 7. Quick Commands

```bash
# Launch Streamlit UI
streamlit run app/main.py

# Run reasoning demo
python examples/run_reasoning_demo.py --question "A train travels 60mph for 2 hours. How far?"

# Run reasoning benchmark (GSM8K mini)
python examples/run_reasoning_benchmark.py --benchmark gsm8k --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --limit 5

# Test LogTokU integration (local Llama)
python examples/test_logtoku_integration.py --query "Write a haiku about uncertainty."
```

---

By following the flow above you can trace any response—from retrieval through uncertainty scoring, adaptive repair, visualization, and benchmarking—making the entire system explainable end-to-end. Feel free to extend this document with project-specific notes as the framework evolves.

