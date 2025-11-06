# Architecture Overview

The system couples real-time token-level uncertainty monitoring with automated self-repair loops that plug into popular agentic frameworks.

## Component Layers
1. **LLM providers (`llm/`)**  
   Abstracts token streaming with logits. The `DeterministicMockLLM` demonstrates how models expose logits for offline tests.

2. **Uncertainty estimators (`uncertainty/`)**  
   `LogTokUEstimator` converts logits into Dirichlet evidence, decomposing uncertainty into aleatoric and epistemic components in a single forward pass.

3. **Repair strategies (`repair/`)**  
   `ConstitutionalRepair` implements rule-based directives that the pipeline injects when uncertainty exceeds adaptive thresholds.

4. **Messaging layer (`messaging/`)**  
   Emits human-readable status updates such as “Moderate uncertainty detected” and “Low confidence - results may vary.”

5. **Agentic pipeline adapters (`pipelines/`)**  
   `UncertaintyAwarePipeline` is the core coordinator. `ControlFlowCoordinator`, `SelfHealingCoordinator`, and `RepairAgentCoordinator` adapt the core to existing frameworks.

6. **Evaluation suite (`evaluation/`)**  
   Provides dataset registry, calibration metrics, and a runner that scores benchmarks with uncertainty tracking.

## Data Flow
```
Prompt → LLM.generate() → (tokens, logits)
      → LogTokU scoring → TokenScore sequence
      → Threshold comparison → (no-op | repair instruction)
      → Self-repair loop → Final GenerationStep
      → StatusMessenger → User-facing updates
```

## Extensibility
- Replace `LLMClient` with a real provider while preserving the `generate` signature.
- Implement new `SelfRepairStrategy` classes, e.g., speculative decoding rollbacks or external tool checks.
- Add richer messaging channels (event streams, dashboards) via pluggable emitters.
