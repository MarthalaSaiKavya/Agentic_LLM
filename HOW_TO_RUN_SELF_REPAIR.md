# How to Run the Self-Repair Agentic System

## Overview

Once LogTokU uncertainty computation is done and thresholds are configured, the self-repair system automatically handles the entire workflow. Here's how to use it:

## Basic Usage: Direct Pipeline

### Step 1: Import Required Components

```python
from token_self_repair.pipelines.base import UncertaintyAwarePipeline, PipelineResult
from token_self_repair.uncertainty.logtoku import LogTokUEstimator
from token_self_repair.repair.constitutional import ConstitutionalRepair
from token_self_repair.messaging.status import StatusMessenger
from token_self_repair.config import ProjectConfig, Thresholds
from token_self_repair.llm.base import LLMClient
```

### Step 2: Set Up Components

```python
# 1. Create LogTokU estimator (uncertainty computation)
estimator = LogTokUEstimator(k=2)

# 2. Create repair strategy (handles repair decisions)
repair_strategy = ConstitutionalRepair()

# 3. Create status messenger (for XAI messages)
messenger = StatusMessenger()

# 4. Configure thresholds (repair triggers when uncertainty > 0.45)
config = ProjectConfig(
    max_self_repairs=2,  # Maximum repair attempts
    thresholds=Thresholds(
        repair_activation_uncertainty=0.45  # Threshold for triggering repair
    )
)

# 5. Get your LLM client (must implement LLMClient interface)
# Options:
# - LlamaProvider (from token_self_repair.llm.llama_provider)
# - OpenAIProvider (from token_self_repair.llm.openai_provider)
# - DeterministicMockLLM (for testing)
llm = YourLLMClient()  # Replace with actual LLM
```

### Step 3: Create and Run Pipeline

```python
# Create the pipeline
pipeline = UncertaintyAwarePipeline(
    llm=llm,
    estimator=estimator,
    strategies=[repair_strategy],
    messenger=messenger,
    config=config
)

# Run it - the system automatically:
# 1. Generates tokens with logits
# 2. Computes uncertainty using LogTokU
# 3. Checks threshold (0.45)
# 4. Triggers repair if uncertainty is high
# 5. Loops until confident or max attempts reached
result: PipelineResult = pipeline.run("What is 2+2?")

# Access results
print("Generated tokens:", result.step.generated_tokens)
print("Final answer:", " ".join(result.step.generated_tokens))
print("Repair attempts:", result.step.repair_attempt)
print("Status messages:", result.messages)

# Check uncertainty scores
if result.step.token_scores:
    final_uncertainty = result.step.token_scores[-1].total_uncertainty
    print(f"Final uncertainty: {final_uncertainty:.2f}")
```

## Complete Example with Mock LLM

```python
from token_self_repair.pipelines.base import UncertaintyAwarePipeline
from token_self_repair.uncertainty.logtoku import LogTokUEstimator
from token_self_repair.repair.constitutional import ConstitutionalRepair
from token_self_repair.messaging.status import StatusMessenger
from token_self_repair.config import ProjectConfig
from token_self_repair.llm.mocks import DeterministicMockLLM

# Set up components
llm = DeterministicMockLLM(
    scripted_responses={"What is 2+2?": ["The", "answer", "is", "4"]},
    vocab_size=8
)
estimator = LogTokUEstimator(k=2)
strategy = ConstitutionalRepair()
messenger = StatusMessenger()
config = ProjectConfig(max_self_repairs=2)

# Create pipeline
pipeline = UncertaintyAwarePipeline(
    llm=llm,
    estimator=estimator,
    strategies=[strategy],
    messenger=messenger,
    config=config
)

# Run it
result = pipeline.run("What is 2+2?")

# Check what happened
print("Answer:", " ".join(result.step.generated_tokens))
print("Repair attempts:", result.step.repair_attempt)
print("Messages:", result.messages)
```

## Using Coordinators (Higher-Level API)

### Option 1: RepairAgentCoordinator (for code repair)

```python
from token_self_repair.pipelines.repair_agent import RepairAgentCoordinator
from token_self_repair.uncertainty.logtoku import LogTokUEstimator
from token_self_repair.repair.constitutional import ConstitutionalRepair
from token_self_repair.llm import LlamaProvider  # or OpenAIProvider

# Set up
llm = LlamaProvider("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
estimator = LogTokUEstimator()
strategy = ConstitutionalRepair()

# Create coordinator (wraps pipeline internally)
coordinator = RepairAgentCoordinator(
    llm=llm,
    estimator=estimator,
    strategies=[strategy]
)

# Use it for code repair
result = coordinator.repair(
    bug_report="Function returns wrong value",
    failing_tests=["assert add(2, 3) == 5"],
    context="def add(a, b): return a - b"
)

print("Fixed code:", " ".join(result.step.generated_tokens))
```

### Option 2: SelfHealingCoordinator (for reasoning tasks)

```python
from token_self_repair.pipelines.self_healing import SelfHealingCoordinator
from token_self_repair.uncertainty.logtoku import LogTokUEstimator
from token_self_repair.repair.constitutional import ConstitutionalRepair
from token_self_repair.llm import LlamaProvider

# Set up
llm = LlamaProvider("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
estimator = LogTokUEstimator()
strategy = ConstitutionalRepair()

# Create coordinator
coordinator = SelfHealingCoordinator(
    llm=llm,
    estimator=estimator,
    strategies=[strategy]
)

# Solve a reasoning task
result = coordinator.solve("A train travels 60 mph for 2 hours. How far?")

print("Answer:", " ".join(result.step.generated_tokens))
print("Confidence check passed:", result.step.final)
```

## How It Works Internally

When you call `pipeline.run(prompt)`, here's what happens automatically:

1. **Token Generation** (`src/token_self_repair/pipelines/base.py:45`)
   - LLM generates tokens with logits
   - Each token is streamed as `TokenLogit` object

2. **Uncertainty Computation** (`src/token_self_repair/pipelines/base.py:91`)
   - For each token, calls `estimator.score(tokens, logits)`
   - LogTokUEstimator computes EU, AU, and total uncertainty
   - Creates `TokenScore` objects with uncertainty metrics

3. **Threshold Check** (`src/token_self_repair/pipelines/base.py:58`)
   - After generation completes, checks final token's `total_uncertainty`
   - Compares against `config.thresholds.repair_activation_uncertainty` (default: 0.45)
   - If uncertainty < threshold → **Finalize** (return result)
   - If uncertainty ≥ threshold → **Trigger Repair**

4. **Repair Execution** (`src/token_self_repair/pipelines/base.py:68-72`)
   - Calls `_execute_repair()` which checks all repair strategies
   - `ConstitutionalRepair.applies()` checks if uncertainty level matches rules
   - `ConstitutionalRepair.repair()` generates corrective instruction
   - Composes new prompt with previous output + repair directive

5. **Refinement Loop** (`src/token_self_repair/pipelines/base.py:76`)
   - Regenerates with refined prompt
   - Repeats steps 1-4 until:
     - Uncertainty drops below threshold (confident), OR
     - Max repair attempts reached (`config.max_self_repairs`)

## Key Configuration Points

### Threshold Configuration (`src/token_self_repair/config.py`)

```python
config = ProjectConfig(
    max_self_repairs=2,  # How many repair attempts allowed
    thresholds=Thresholds(
        repair_activation_uncertainty=0.45,  # When to trigger repair
        high_confidence=0.8,      # For uncertainty level classification
        moderate_confidence=0.6,  # For uncertainty level classification
        low_confidence=0.4        # For uncertainty level classification
    )
)
```

### Repair Strategy Rules (`src/token_self_repair/repair/constitutional.py`)

The `ConstitutionalRepair` strategy has default rules, but you can customize:

```python
from token_self_repair.types import UncertaintyLevel

custom_strategy = ConstitutionalRepair(
    rules={
        UncertaintyLevel.LOW: [
            "Your custom repair instruction here",
            "Another instruction for low confidence"
        ],
        UncertaintyLevel.MODERATE: [
            "Custom instruction for moderate uncertainty"
        ]
    }
)
```

## Real-World Example with Llama

```python
from token_self_repair.pipelines.base import UncertaintyAwarePipeline
from token_self_repair.uncertainty.logtoku import LogTokUEstimator
from token_self_repair.repair.constitutional import ConstitutionalRepair
from token_self_repair.messaging.status import StatusMessenger
from token_self_repair.config import ProjectConfig
from token_self_repair.llm import LlamaProvider, load_llama

# Load Llama model
llm = load_llama("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Set up self-repair components
estimator = LogTokUEstimator(k=2)
strategy = ConstitutionalRepair()
messenger = StatusMessenger()
config = ProjectConfig(max_self_repairs=3)

# Create pipeline
pipeline = UncertaintyAwarePipeline(
    llm=llm,
    estimator=estimator,
    strategies=[strategy],
    messenger=messenger,
    config=config
)

# Run with a question
result = pipeline.run("Explain quantum computing in simple terms", max_tokens=100)

# Check results
print("=" * 50)
print("FINAL ANSWER:")
print(" ".join(result.step.generated_tokens))
print("=" * 50)
print(f"Repair attempts: {result.step.repair_attempt}")
print(f"Final uncertainty: {result.step.token_scores[-1].total_uncertainty:.3f}")
print("=" * 50)
print("Status messages during execution:")
for msg in result.messages:
    print(f"  - {msg}")
```

## What Happens When Uncertainty is High?

If the final token's uncertainty exceeds the threshold (0.45):

1. **Repair Triggered**: `ConstitutionalRepair` generates an instruction like:
   ```
   "A self-repair was triggered:
   - Detected uncertainty level: MODERATE
   - Action: Clarify assumptions and provide alternative interpretations."
   ```

2. **Prompt Refinement**: The system creates a new prompt:
   ```
   [Original Question]
   
   The assistant previously produced the following draft response:
   [Previous output]
   
   Revise the answer while following this directive:
   [Repair instruction]
   ```

3. **Regeneration**: LLM generates again with the refined prompt

4. **Re-evaluation**: Uncertainty is checked again - if still high, another repair attempt is made (up to `max_self_repairs`)

## Summary

**To run the self-repair system:**

1. ✅ LogTokU is already integrated - just create `LogTokUEstimator()`
2. ✅ Thresholds are configured in `ProjectConfig` - default is 0.45
3. ✅ Create `UncertaintyAwarePipeline` with all components
4. ✅ Call `pipeline.run(prompt)` - **everything else is automatic!**

The system handles:
- Token generation → Uncertainty computation → Threshold check → Repair decision → Refinement loop

You just provide the prompt and get back a refined answer with uncertainty tracking!

