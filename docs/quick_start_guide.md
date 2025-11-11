# Quick Start Guide

## Overview

This project combines LogTokU uncertainty estimation with RepairAgent's autonomous repair capabilities to create a novel, state-of-the-art uncertainty-guided hierarchical program repair system.

## Architecture Components

```
┌─────────────────┐
│  Llama Models   │  LogTokU-instrumented
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Uncertainty     │  Token → Line → Method
│ Engine          │  AU + EU decomposition
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Strategy        │  Exploration vs Refinement
│ Selector        │  Based on AU/EU
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RepairAgent     │  Patch generation + testing
│ Core            │  Uncertainty-guided
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Patch Ranker    │  Novel UncertaintyScore
│                 │  α×confidence + β×tests + γ×diversity
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │  Heatmaps + Trajectories
│ & XAI           │  Interactive explanations
└─────────────────┘
```

## Current Progress

Track progress in [implementation_plan.md](implementation_plan.md)

## Phase 1 - Start Here

### Step 1: Setup Llama Models
```bash
cd /home/dpendya/Documents/dlba/logtoku
conda activate logtoku

# Verify models are downloaded
ls models/
# Should see: meta-llama/Llama-2-7b-chat-hf, Llama-3-8B-Instruct, etc.
```

### Step 2: Test LogTokU
```bash
cd SenU
python generate.py --model_name llama2_chat_7B --gene 1 --mode one_pass --gpuid 0
```

Want to exercise the hosted OpenAI path instead of a local model? Export
`OPENAI_API_KEY` and run the consolidated example with the OpenAI backend flag:

```bash
python examples/test_logtoku_integration.py --backend openai --openai-model gpt-4o-mini
```

### Step 3: Install Project Dependencies
```bash
cd /home/dpendya/Documents/dlba
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Step 4: Run Initial Tests
```bash
pytest tests/
```

## Key Directories

```
dlba/
├── logtoku/                 # LogTokU implementation (already installed)
│   ├── models/              # Llama models
│   ├── SenU/                # Uncertainty estimation code
│   └── DynDecoding/         # Dynamic decoding experiments
│
├── RepairAgent/             # Autonomous repair agent (already installed)
│   └── repair_agent/        # Core FSM logic
│
├── src/token_self_repair/   # Our unified framework
│   ├── uncertainty/         # LogTokU integration + aggregation
│   ├── repair/              # Strategy selection + mutations
│   ├── pipelines/           # RepairAgent integration
│   ├── evaluation/          # Benchmarks + metrics
│   └── visualization/       # Heatmaps + explanations
│
├── docs/
│   ├── implementation_plan.md    # MAIN CHECKLIST (start here!)
│   ├── architecture.md           # System design
│   └── evaluation_plan.md        # Metrics and benchmarks
│
└── tests/                   # Unit tests
```

## Implementation Sequence

Follow the phases in order:

1. **Phase 1**: Core Infrastructure (LogTokU + Multi-granularity)
2. **Phase 2**: RepairAgent Integration (Adapter + Uncertainty injection)
3. **Phase 3**: Dynamic Strategies (Exploration vs Refinement)
4. **Phase 4**: Patch Ranking (Novel UncertaintyScore metric)
5. **Phase 5**: Evaluation (4 benchmarks + ablation studies)
6. **Phase 6**: Visualization (Heatmaps + trajectories + XAI)
7. **Phase 7**: Documentation (Paper + reproducibility package)

## Daily Workflow

1. Open `docs/implementation_plan.md`
2. Pick the next unchecked task
3. Implement in appropriate `src/token_self_repair/` module
4. Write unit test in `tests/`
5. Check off task in implementation plan
6. Commit progress
7. Repeat

## Expected Outcomes

### Quantitative
- 15-25% improvement in Defects4J fix rate
- 30-40% fewer patches per successful fix
- AUROC > 0.75 for uncertainty calibration
- ECE < 0.08 for calibration quality

### Qualitative
- Heatmaps identify bugs with >70% accuracy
- Strategy selection matches expert analysis >80%
- Patch rankings correlate with success (ρ > 0.6)

## Resources

- **LogTokU Paper**: https://arxiv.org/abs/2502.00290
- **RepairAgent Paper**: https://arxiv.org/abs/2403.17134
- **Models**: meta-llama/Llama-2-7b-chat-hf, Llama-3-8B-Instruct
- **Benchmarks**: GSM8K, TruthfulQA, Defects4J, HumanEval

## Need Help?

- Check existing code in `src/token_self_repair/` for examples
- Reference LogTokU implementation in `logtoku/SenU/`
- Reference RepairAgent logic in `RepairAgent/repair_agent/`
- Read architecture in `docs/architecture.md`

## Quick Commands

```bash
# Install core + UI/Llama/OpenAI extras
pip install -e ".[ui,openai,llama]"

# Activate environment
cd /home/dpendya/Documents/dlba
source venv/bin/activate  # or: conda activate logtoku

# Run tests
pytest tests/test_uncertainty.py
pytest tests/test_self_repair.py

# Run single benchmark
python -m token_self_repair.evaluation.runner --benchmark gsm8k

# Run full evaluation
python -m token_self_repair.evaluation.runner --all

# Generate visualizations
python -m token_self_repair.visualization.heatmap --input results.json

# Launch the Streamlit dashboard (chat + benchmarks + program repair tabs)
streamlit run app/main.py
```

## Next Steps

1. Open `docs/implementation_plan.md`
2. Start with Phase 1.1: LogTokU Integration with Llama Models
3. Check off tasks as you complete them
4. Track overall progress at the bottom of the plan

Good luck with the implementation!
