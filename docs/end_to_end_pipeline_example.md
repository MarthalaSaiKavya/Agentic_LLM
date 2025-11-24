# End-to-End Pipeline Walkthrough

This document walks through a single `UncertaintyAwarePipeline` run and shows how
token-level uncertainty propagates into repair triggers, strategy selection, and
final messaging. All numbers are illustrative but mirror the exact logic coded
in `src/token_self_repair`.

## 1. Initial Configuration

| Component | Setting | Notes |
| --- | --- | --- |
| `ProjectConfig.max_self_repairs` | `2` | Allows at most two repair passes. |
| `Thresholds.repair_activation_uncertainty` | `0.45` | Anything above this triggers a repair loop. |
| LLM client | `LlamaProvider("TinyLlama/TinyLlama-1.1B-Chat-v1.0")` | Streams tokens + logits. |
| Estimator | `LogTokUEstimator(k=2)` | Produces per-token EU, AU, entropy, total uncertainty. |
| Strategies | `LineLevel`, `MethodLevel`, `TestGuided` | Ordered dynamically by EU vs AU signal. |
| Messenger | `StatusMessenger` | Emits human-readable updates every time the pipeline observes new info. |

Prompt:

```
Explain how quicksort works on the array [8, 3, 5].
```

## 2. First Generation Pass

The LLM streams five illustrative tokens:

| Token | Max logit | Probability | Entropy | Aleatoric (AU) | Epistemic (EU) | Total (LogTokU) | Level |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `"Quicksort"` | 2.11 | 0.47 | 1.22 | 0.31 | 0.22 | **0.07** | High confidence |
| `" picks"` | 1.42 | 0.36 | 1.45 | 0.46 | 0.51 | **0.24** | Moderate |
| `" 3"` | 0.71 | 0.28 | 1.61 | 0.58 | 0.63 | **0.37** | Moderate |
| `" as"` | 0.04 | 0.17 | 1.89 | 0.66 | 0.74 | **0.49** | Low |
| `" pivot"` | -0.33 | 0.10 | 2.04 | 0.71 | 0.82 | **0.58** | Low |

Uncertainty summary (`_compute_uncertainty_profile`):

```
avg_total = 0.35
avg_eu    = 0.58
avg_au    = 0.54
max_total = 0.58
```

- The final token has `total_uncertainty = 0.58`, which is > `0.45`, so
  `repair_activation_uncertainty` triggers a repair pass.
- Because `avg_eu (0.58) ≥ avg_au (0.54)`, `_order_strategies` prioritizes
  method-centric strategies (method → test → line).

## 3. Repair Attempt #1

1. **Strategy selection**  
   - `MethodLevelRepairStrategy.applies` returns `True` (`repair_attempt == 0`
     but method strategy only needs `>= 1`, so the first pass still routes to
     `LineLevel`. When repair attempt increments later, method strategy is allowed.)
   - `LineLevelRepairStrategy` inspects the uncertainty map (token-to-line
     aggregation) and finds that the sentence containing `"as pivot"` (line 4)
     has total uncertainty `0.56`. It produces the instruction:

     > Focus on line 4 which shows high uncertainty (score=0.56). Rewrite this
     > line to clarify the pivot choice and the recursive calls. Current line:
     > `As pivot, explain recursion poorly`.

2. **Messenger updates**  
   - Token-level notifications already logged “Low confidence – results may vary”
     for the `"as"` and `"pivot"` tokens.
   - `notify_repair` emits “Low confidence – running corrective pass” with the
     line-focused instruction.

3. **Prompt composition**  
   `_compose_repair_prompt` appends the draft output and instruction to the original
   question, producing a revised prompt for attempt #1.

## 4. Second Generation Pass

New token stream (abridged):

| Token | Total Uncertainty |
| --- | --- |
| `"Quicksort"` | 0.05 |
| `" selects"` | 0.18 |
| `" 3"` | 0.27 |
| `" as"` | 0.34 |
| `" pivot"` | 0.41 |
| `" and"` | 0.29 |
| `" recurses"` | 0.22 |
| `" on"` | 0.20 |
| `" [3]"` | 0.17 |
| `" and"` | 0.16 |
| `" [5,8]"` | 0.19 |

- Final token uncertainty: `0.19`, now **below** the 0.45 activation threshold.
- Average total uncertainty dropped to `0.23`, EU vs AU signals converge and no
  additional passes are needed.
- `notify_completion` announces “High confidence response – Generated 11 tokens
  across 2 pass(es).”

## 5. Result Artifacts

### Final Answer (joined tokens)

> Quicksort selects 3 as pivot and recurses on [3] and [5,8], concatenating
> left + pivot + right to yield [3,5,8].

### Strategy & Repair Notes

| Attempt | Strategy | Instruction Summary |
| --- | --- | --- |
| 0 (initial) | — | Raw generation; no instruction yet. |
| 1 | Line-level | Rewrite line describing pivot rationale and recursive flow. |

### Messenger History

1. High confidence response – “Quicksort”
2. Moderate uncertainty detected – “ picks”
3. Moderate uncertainty detected – “ 3”
4. Low confidence – results may vary – “ as”
5. Low confidence – results may vary – “ pivot”
6. Low confidence – running corrective pass – line-level instruction
7. High confidence response – completion summary

### Hotspot Map Snapshot

| Kind | Identifier | Score | Meaning |
| --- | --- | --- | --- |
| Line | `4` | `0.56` | Pivot explanation sentence flagged for rewrite. |
| Method | `<document>` | `0.35` | Overall paragraph still moderately uncertain before repair. |

## 6. Takeaways

- **Thresholding:** `repair_activation_uncertainty` is consulted on each pass
  using the most recent token; average uncertainty only influences strategy
  ordering.
- **Strategy ordering:** Higher epistemic than aleatoric uncertainty prioritizes
  method-level or exploration strategies; the opposite favors line/test focus.
- **Repair prompts:** Each instruction is appended to the original question to
  preserve context while nudging the LLM toward targeted revisions.
- **Messaging:** Every significant change (token-level status, repair trigger,
  completion) is surfaced through `StatusMessenger`, making it easy to stream
  updates into the Streamlit UI or logs.

This single walkthrough mirrors the runtime behavior you can observe by wiring a
real LLM client into `UncertaintyAwarePipeline` (or by running
`examples/test_logtoku_integration.py` followed by a repair coordinator).

