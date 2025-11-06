# Evaluation Plan

## Benchmarks
- **Multi-step reasoning:** GSM8K-lite samples from `evaluation.datasets.gsm8k_samples`.
- **Code repair:** HumanEval-lite snippets mirroring RepairAgent workloads.
- **Factual robustness:** Extend by registering TruthfulQA or BioASQ loaders with the dataset registry.

## Metrics
| Metric | Description | Expected Signal |
| --- | --- | --- |
| AUROC (uncertainty vs. error) | Binary discrimination between correct and incorrect generations. | >0.60 in early prototypes, improving with better estimators. |
| Expected Calibration Error | Closeness of uncertainty-derived probabilities to empirical accuracy. | <0.08 for well-calibrated systems. |
| Task-Specific Score | e.g., exact match for reasoning or pass@1 for code generation. | 15-25% uplift after self-repair loops. |
| Latency Overhead | Wall-clock delta relative to raw agent workflow. | <20% penalty due to single-pass LogTokU computation. |
| User Trust Correlation | Survey-based correlation between messaging and perceived reliability. | Positive correlation ≥0.4. |

## Procedure
1. Instantiate a `pipeline_factory` that returns `UncertaintyAwarePipeline` objects wired to the target agentic workflow.
2. Use `EvaluationRunner.run(benchmark_name)` to iterate through samples, capturing predictions, uncertainties, and status telemetry.
3. Compute downstream metrics via the benchmark’s metric callback; derive calibration using `expected_calibration_error`.
4. Compare against baselines with uncertainty disabled to measure repair-induced gains.
5. Conduct user studies by logging `StatusMessenger.history` during interactive sessions and correlating with user confidence ratings.
