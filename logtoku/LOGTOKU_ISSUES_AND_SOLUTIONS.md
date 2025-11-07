# LogTokU Implementation Issues and Solutions

## Problem Statement

After testing LogTokU on the queries "Who is Deepak Sai Pendyala" and "What is Deep Learning", we identified several issues with the current implementation.

## Issues Found

### 1. **EU Calculation Fails with Negative Logits**

**Problem**: The original EU formula is:
```python
mean_scores = top_k / (np.sum(np.maximum(0, top_values)) + top_k)
```

When all logits are negative (common in GPT-2), `np.maximum(0, top_values)` returns 0, making EU always = 1.0, which indicates maximum uncertainty even when the model is confident.

**Evidence**: In our tests, GPT-2 produces logits like -123.625, -130.0, etc., causing EU to always be 1.0.

### 2. **AU Calculation Issues with Very Negative Values**

**Problem**: AU uses `digamma(alpha + 1)` where `alpha` are the logits. When logits are very negative (e.g., -123.625), `digamma(-122.625)` can produce NaN or invalid results.

**Evidence**: We observed NaN values in AU calculations for some tokens.

### 3. **Model-Specific Logit Ranges**

**Problem**: The original LogTokU implementation was tested with Llama models, which may produce logits in a different range than GPT-2. GPT-2's logits are often very negative.

## Solutions Implemented

### Solution 1: Fixed EU Calculation

Instead of using raw logits with `max(0, ...)`, we normalize logits to preserve relative relationships:

```python
# Shift to make top value = 0 (preserves relative differences)
top_values_shifted = top_values - top_values.max()
# Convert to positive space using exponential
exp_top = np.exp(top_values_shifted)
eu = top_k / (np.sum(exp_top) + top_k)
```

This ensures EU properly reflects uncertainty even when all logits are negative.

### Solution 2: Fixed AU Calculation

Shift logits to be positive before applying digamma:

```python
# Make all values positive while preserving relationships
top_values_for_au = top_values - np.min(top_values) + 1.0
# Then apply digamma
```

### Solution 3: Comparison Results

With the fixed implementation:

- **"Who is Deepak Sai Pendyala"**: LogTokU = 0.2681
- **"What is Deep Learning"**: LogTokU = 0.2813

The difference is now visible, though still small. This suggests:
1. Both queries have similar uncertainty (which may be correct for GPT-2)
2. GPT-2 may not be the ideal model for this test
3. The model itself may have similar uncertainty levels for both types of queries

## Root Cause Analysis

### Why This Happens

1. **Different Model Architectures**: Llama models (used in original paper) vs GPT-2 produce logits in different ranges
2. **Logit Normalization**: The original implementation assumes logits will have some positive values or be in a specific range
3. **Formula Assumptions**: The EU formula assumes that `max(0, top_values)` will capture meaningful information, which fails when all values are negative

### Expected Behavior

According to the paper:
- **Higher LogTokU** = Higher uncertainty (model less confident)
- **Lower LogTokU** = Lower uncertainty (model more confident)

However, note that:
- High confidence ≠ correctness (models can be confidently wrong)
- Low uncertainty on incorrect information indicates a calibration problem

## Recommendations

### 1. Use Appropriate Models

The original implementation was designed for Llama models. For best results:
- Use Llama-2 or Llama-3 models
- Or use models that produce logits in a similar range

### 2. Model-Specific Adjustments

If using different models (like GPT-2):
- Use the fixed implementation (`test_queries_fixed.py`)
- Or normalize logits to the expected range before calculation

### 3. Interpretation

When interpreting results:
- **Small differences** in LogTokU may indicate the model has similar uncertainty levels
- **Check individual token metrics** to understand where uncertainty occurs
- **Compare with entropy** to validate results
- **Consider the model's calibration** - low uncertainty on wrong answers is a model issue, not a LogTokU issue

## Files

- `test_queries.py` - Original implementation (has issues with negative logits)
- `test_queries_fixed.py` - Fixed implementation (handles negative logits properly)
- `debug_real_logits.py` - Diagnostic tool to examine actual logits
- `diagnose_logtoku.py` - Comprehensive diagnostic tests

## Conclusion

**Is LogTokU working properly?**

- The **concept** is sound: combining EU and AU provides useful uncertainty estimates
- The **original implementation** has issues with very negative logits (common in GPT-2)
- The **fixed implementation** handles negative logits correctly
- **Results may vary** depending on the model used - the original was tested with Llama models
- **Small differences** in uncertainty scores may be expected when the model genuinely has similar uncertainty levels for different queries

For production use, we recommend:
1. Using the fixed implementation
2. Testing with the model types used in the original paper (Llama models)
3. Validating results against entropy and other uncertainty metrics
4. Understanding that LogTokU measures uncertainty, not correctness

