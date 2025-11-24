# Self-Repair Walkthrough: “What is the conclusion of this project?”

This document captures the Streamlit run described in the transcript above and
explains how the uncertainty-aware pipeline behaved from retrieval through
confidence signaling (no repair was required because the model remained within
thresholds). Use it as a template for documenting future sessions.

## 1. Prompt & Goal

- **User question:** “What is the conclusion of this project?”
- **Context:** Knowledge base contained the PDF “DLBA_New.pdf” covering the
  project write-up. No custom instructions beyond the default assistant prompt.

## 2. Retrieval Stage

The RAG pipeline performed similarity search (`top_k=4`) against the in-memory
vector store. Thanks to the new deduplication logic in `app/rag.py`, two unique
chunks were surfaced:

| Doc | Score | Source | Summary |
| --- | --- | --- | --- |
| 1 | 0.16 | `DLBA_New.pdf` | Defines epistemic vs aleatoric uncertainty and how aggregation turns token metrics into explainable visuals. |
| 2 | 0.14 | `DLBA_New.pdf` | Describes multi-level repair strategies, including line/method escalation and patch ranking. |

No redundant snippets were injected into the prompt, so the LLM received a clean
context block before generating.

## 3. Generation Output

The answer emphasized that the project’s conclusion isn’t explicitly written in
the source, but the system’s purpose can be inferred: aggregate uncertainty,
surface hotspots via the UI, and apply adaptive repair. Because the model did
not need to invent unsupported claims, the initial draft was accepted as-is.

Excerpt:

> “Based on the provided references, the conclusion of this project is not
> explicitly stated. However, it can be inferred that the project aims to
> develop a system that addresses uncertainty in generated content by providing
> a framework for aggregation, explainability, and adaptive repair…”

## 4. Uncertainty Metrics

LogTokU analysis produced the following averages:

| Metric | Value |
| --- | --- |
| Avg EU | 0.041 |
| Avg AU | 0.673 |
| Avg LogTokU | **0.028** |
| Avg Entropy | 0.592 |

- **Threshold:** 0.15 (configured via sidebar slider).
- **Status banner:** “Uncertainty status: LOW — Avg LogTokU 0.03 vs threshold 0.15”.
- Since `avg_logtoku < threshold`, neither `low_confidence` nor repair logic
  triggered. The Streamlit UI showed a green success badge instead of a warning.

## 5. Token/Hotspot Inspection

Even though the run was high-confidence, the UI still rendered:

- **Hotspot timeline:** essentially flat, max total uncertainty < 0.03 across all
  lines.
- **Token map:** highlighted a few words (“framework”, “system”) with slightly
  higher AU because the response repeated them, but the magnitudes remained
  below 0.07.

These panels confirmed there were no localized spikes requiring intervention.

## 6. Self-Repair Analysis

Because thresholds were not breached:

- `UncertaintyAwarePipeline.run` stayed in its initial pass.
- `LineLevelRepairStrategy`, `MethodLevelRepairStrategy`, and
  `TestGuidedRepairStrategy` were never invoked.
- Messenger history contained only the automatic “High confidence response”
  entry—no “repair” or “low confidence” messages were logged.

Had any token exceeded the 0.15 threshold, the pipeline would have:

1. Logged “Low confidence – running corrective pass”.
2. Selected the most appropriate strategy (line/method/test) based on avg EU vs
   AU.
3. Fed a targeted instruction back into the LLM and recomputed uncertainty on the
   revised draft.

## 7. Takeaways

- The new UI enhancements (explicit status banner + deduped context list) make
  it easy to see that a run completed confidently.
- Even when no repair is needed, the system still exposes the evidence (token
  map and metrics) so reviewers understand why the answer was accepted.
- If you need to demonstrate the full repair loop, rerun a question with a much
  lower threshold (e.g., 0.05) or use a prompt known to stress the model; the
  same logging and reporting structure will record each attempt.

This concludes the documented run. Store this file alongside other experiment
notes (`docs/`) to keep a clear history of how the pipeline behaves for real
queries.

