"""
Quick example: How to run the self-repair agentic system.

This demonstrates the complete workflow after LogTokU and thresholds are set up.
"""

from token_self_repair.pipelines.base import UncertaintyAwarePipeline
from token_self_repair.uncertainty.logtoku import LogTokUEstimator
from token_self_repair.repair.constitutional import ConstitutionalRepair
from token_self_repair.messaging.status import StatusMessenger
from token_self_repair.config import ProjectConfig, Thresholds
from token_self_repair.llm.mocks import DeterministicMockLLM

# ============================================================================
# STEP 1: Set up all components
# ============================================================================

# 1. LLM Client (generates tokens with logits)
#    In production, use: LlamaProvider or OpenAIProvider
#    For testing, use: DeterministicMockLLM
llm = DeterministicMockLLM(
    scripted_responses={
        "What is 2+2?": ["The", "answer", "is", "4", "."]
    },
    vocab_size=8
)

# 2. LogTokU Estimator (computes uncertainty from logits)
estimator = LogTokUEstimator(k=2)

# 3. Repair Strategy (decides what to do when uncertainty is high)
repair_strategy = ConstitutionalRepair()

# 4. Status Messenger (tracks XAI messages)
messenger = StatusMessenger()

# 5. Configuration (thresholds and limits)
config = ProjectConfig(
    max_self_repairs=2,  # Maximum repair attempts
    thresholds=Thresholds(
        repair_activation_uncertainty=0.45  # Trigger repair if uncertainty > 0.45
    )
)

# ============================================================================
# STEP 2: Create the pipeline
# ============================================================================

pipeline = UncertaintyAwarePipeline(
    llm=llm,
    estimator=estimator,
    strategies=[repair_strategy],
    messenger=messenger,
    config=config
)

# ============================================================================
# STEP 3: Run it! (Everything is automatic from here)
# ============================================================================

print("=" * 60)
print("Running Self-Repair Pipeline")
print("=" * 60)
print()

prompt = "What is 2+2?"
print(f"Prompt: {prompt}")
print()

# This single call handles:
# - Token generation with logits
# - LogTokU uncertainty computation
# - Threshold checking (0.45)
# - Repair triggering if needed
# - Refinement loop until confident
result = pipeline.run(prompt, max_tokens=50)

# ============================================================================
# STEP 4: Check results
# ============================================================================

print("=" * 60)
print("RESULTS")
print("=" * 60)
print()

# Final answer
answer = " ".join(result.step.generated_tokens)
print(f"Final Answer: {answer}")
print()

# Repair information
print(f"Repair Attempts: {result.step.repair_attempt}")
print(f"Final Status: {'Finalized' if result.step.final else 'Incomplete'}")
print()

# Uncertainty information
if result.step.token_scores:
    final_score = result.step.token_scores[-1]
    print("Final Token Uncertainty:")
    print(f"  - Total Uncertainty: {final_score.total_uncertainty:.3f}")
    print(f"  - Epistemic (EU): {final_score.epistemic:.3f}")
    print(f"  - Aleatoric (AU): {final_score.aleatoric:.3f}")
    print(f"  - Level: {final_score.level.name}")
    print()

# Status messages (XAI)
print("Status Messages:")
for i, msg in enumerate(result.messages, 1):
    print(f"  {i}. {msg}")
print()

print("=" * 60)
print("Complete!")
print("=" * 60)

# ============================================================================
# Alternative: Using a Coordinator (Higher-level API)
# ============================================================================

print("\n" + "=" * 60)
print("Alternative: Using SelfHealingCoordinator")
print("=" * 60)
print()

from token_self_repair.pipelines.self_healing import SelfHealingCoordinator

# Coordinator wraps the pipeline with domain-specific logic
coordinator = SelfHealingCoordinator(
    llm=llm,
    estimator=estimator,
    strategies=[repair_strategy],
    messenger=StatusMessenger(),
    config=config
)

# Solve a reasoning task
reasoning_result = coordinator.solve("A train travels 60 mph for 2 hours. How far?")

print(f"Answer: {' '.join(reasoning_result.step.generated_tokens)}")
print(f"Confidence check passed: {reasoning_result.step.final}")

