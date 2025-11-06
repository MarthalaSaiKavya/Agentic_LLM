# Token-Level Uncertainty Driven Self-Repair

This project implements an open-source framework that infuses agentic LLM workflows with token-level uncertainty monitoring, automated self-repair, and transparent user messaging. The system integrates Logits-induced Token Uncertainty (LogTokU), adaptive confidence thresholding, and iterative refinement loops inspired by constitutional AI techniques.

## Project Goals
- Detect low-confidence tokens during generation without resorting to multi-sampling.
- Trigger self-repair strategies tailored to the host agentic framework.
- Communicate uncertainty and repair status to downstream users in real time.
- Supply an evaluation harness across reasoning, coding, and factuality benchmarks.

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
pip install -e .
pytest
```

Refer to `docs/architecture.md` for a deep dive into components and data flow.
