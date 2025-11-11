# Token-Level Uncertainty Driven Self-Repair

This project implements an open-source framework that infuses agentic LLM workflows with token-level uncertainty monitoring, automated self-repair, and transparent user messaging. The system integrates Logits-induced Token Uncertainty (LogTokU), adaptive confidence thresholding, and iterative refinement loops inspired by constitutional AI techniques.

## Novel Contributions

1. **Uncertainty-Driven Patch Ranking**: Novel metric combining token-level confidence scores with test execution feedback
2. **Dynamic Strategy Selection**: Epistemic uncertainty triggers search/exploration, aleatoric uncertainty triggers refinement
3. **Multi-Granularity Healing**: Operates at token, statement, method, and test levels simultaneously
4. **Unified Framework**: Single architecture handles both reasoning tasks (GSM8K) and program repair (Defects4J)
5. **Real-time Explainability**: Uncertainty visualization showing WHERE and WHY the model is uncertain

## Project Goals
- Detect low-confidence tokens during generation without resorting to multi-sampling.
- Trigger self-repair strategies tailored to the host agentic framework.
- Communicate uncertainty and repair status to downstream users in real time.
- Supply an evaluation harness across reasoning, coding, and factuality benchmarks.

## Getting Started

See the comprehensive [Implementation Plan](docs/implementation_plan.md) for the complete roadmap with checkboxes to track progress.

## Repository Layout
```
src/token_self_repair/
├── llm/               # LLM client abstractions and mock providers
├── uncertainty/       # Token-level uncertainty estimators (LogTokU, calibration utilities)
├── repair/            # Repair strategies (constitutional rules, sampling backtracking)
├── pipelines/         # Integrations for ControlFlow, Self-Healing LLM Pipeline, RepairAgent
├── messaging/         # Status messaging and telemetry emitters
├── evaluation/        # Dataset registry, metric computation, benchmark runner
├── utils/             # Shared utilities (logging, token helpers)
└── config.py          # Central configuration and thresholds
```

The `docs/` directory explains architectural decisions and evaluation protocols, while `tests/` contains verification for uncertainty decomposition and repair loop control flow.

## Quick Start
```bash
# install the Streamlit UI + OpenAI + local Llama extras
pip install -e ".[ui,openai,llama]"
pytest
```

Optional extras:

- `pip install -e ".[ui]"` – Streamlit dashboard + visualization stack (pandas/altair)
- `pip install -e ".[openai]"` – OpenAI + langchain-openai + ddgs search fallback
- `pip install -e ".[llama]"` – torch/transformers/sentence-transformers for local models

Refer to `docs/architecture.md` for a deep dive into components and data flow.

### Visualization add-ons

The Streamlit chat view now exposes:

- **Hotspot timeline** – line-by-line uncertainty plot derived from LogTokU aggregates
- **Token map** – sortable table of tokens with epistemic/aleatoric breakdowns
- **Message history pane** – quickly skim recent prompts/answers without scrolling the chat feed

Each panel logs to the console so you can trace uncertainty spikes back to the underlying tokens/lines.

### Program repair benchmarks

The new _Program Repair_ tab runs the multi-level repair coordinator (line/method/test strategies) on mini versions of the Repair, Defects4J, and GitBugs datasets. Metrics include pass rate, pass@1, and latency, and every attempt logs its selected strategy plus messenger history for fast debugging.

### Using OpenAI logprobs

Set your API key and run the example with `--backend openai` to inspect token-level
logprobs returned by models such as `gpt-4o-mini`:

```bash
export OPENAI_API_KEY="sk-..."
python examples/test_logtoku_integration.py --backend openai --openai-model gpt-4o-mini
```

The script prints the same LogTokU-derived uncertainty metrics and hotspot summaries as
the Hugging Face path, making it easy to compare OpenAI responses with local Llama runs.
