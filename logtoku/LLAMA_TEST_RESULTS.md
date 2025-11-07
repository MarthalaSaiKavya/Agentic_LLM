# LogTokU Test Results with Llama Models

## Test Configuration

- **Model**: TinyLlama-1.1B-Chat-v1.0 (Llama-based, publicly available)
- **Queries**: 
  1. "Who is Deepak Sai Pendyala"
  2. "What is Deep Learning"
- **Max Tokens**: 30
- **Quantization**: 4-bit (enabled)

## Key Findings

### ✅ Original Implementation Works with Llama Models!

The original LogTokU implementation works correctly with Llama-based models, unlike GPT-2:

**Original Implementation Results:**

| Query | Avg LogTokU | Avg EU | Avg AU | Avg Entropy |
|-------|-------------|--------|--------|-------------|
| Who is Deepak Sai Pendyala | **0.0384** | 0.0576 | 0.6653 | 1.0353 |
| What is Deep Learning | **0.0398** | 0.0593 | 0.6699 | 1.2614 |

**Observations:**
- EU values are reasonable (0.0576, 0.0593) - not stuck at 1.0 like with GPT-2
- AU values are consistent (~0.67)
- LogTokU values are meaningful and show slight differences
- "What is Deep Learning" shows slightly higher uncertainty (0.0398 vs 0.0384)

### Fixed Implementation (for comparison):

| Query | Avg LogTokU | Avg EU | Avg AU | Avg Entropy |
|-------|-------------|--------|--------|-------------|
| Who is Deepak Sai Pendyala | 0.2428 | 0.6284 | 0.3930 | 1.0353 |
| What is Deep Learning | 0.2624 | 0.6254 | 0.4243 | 1.2614 |

**Note**: Fixed implementation shows different absolute values but similar relative patterns.

## Conclusions

1. **Original implementation is correct** - The issue was model-specific (GPT-2's very negative logits), not an implementation bug

2. **Llama models work well** - The original implementation was designed for Llama models and works as expected

3. **Uncertainty estimates are meaningful** - Both queries show similar uncertainty levels, with "What is Deep Learning" slightly higher

4. **Model matters** - GPT-2 requires the fixed implementation, while Llama models work with the original

## Using Official Meta Llama Models

To test with official Meta Llama models (Llama-2, Llama-3):

1. **Request Access**:
   - Visit: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   - Click "Request access" and follow the instructions
   - Wait for approval (usually takes a few hours to a day)

2. **Once Approved**, run:
   ```bash
   cd /home/dpendya/Documents/dlba/logtoku
   source venv/bin/activate
   python test_llama.py --model meta-llama/Llama-2-7b-chat-hf \
                        --queries "Who is Deepak Sai Pendyala" "What is Deep Learning" \
                        --max_tokens 30 --quantize
   ```

3. **Alternative Models** (no access required):
   - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (tested above)
   - `microsoft/phi-2` (small, publicly available)
   - Other open-source Llama variants

## Recommendations

1. **Use Llama models** for production/testing with LogTokU
2. **Use original implementation** when working with Llama models
3. **Use fixed implementation** only for models with very negative logits (like GPT-2)
4. **Request Meta Llama access** for best results matching the original paper

## Next Steps

1. Request access to Meta Llama models on Hugging Face
2. Test with larger Llama models (7B, 13B) for comparison
3. Compare results across different model sizes
4. Validate uncertainty estimates with ground truth data

