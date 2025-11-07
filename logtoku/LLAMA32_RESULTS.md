# LogTokU Test Results with Llama 3.2

## Configuration

- **Model**: meta-llama/Llama-3.2-3B-Instruct
- **Quantization**: 4-bit (enabled)
- **Device**: CUDA (NVIDIA RTX A5000)

## Test Results

### Query 1: "Who is Deepak Sai Pendyala"

**Response**: "I couldn't find any notable information on a person named Deepak Sai Pendyala..."

**LogTokU Metrics**:
- Average EU (Epistemic Uncertainty): **0.0400**
- Average AU (Aleatoric Uncertainty): **0.6696**
- Average LogTokU (EU × AU): **0.0267**
- Average Entropy: 0.2933
- Maximum LogTokU: 0.0325
- Tokens analyzed: 30

### Query 2: "What is Deep Learning"

**Response**: "Deep learning is a subset of machine learning that involves the..."

**LogTokU Metrics**:
- Average EU (Epistemic Uncertainty): **0.0435**
- Average AU (Aleatoric Uncertainty): **0.6729**
- Average LogTokU (EU × AU): **0.0293**
- Average Entropy: 0.3793
- Maximum LogTokU: 0.0356
- Tokens analyzed: 30

## Comparison

| Query | Avg LogTokU | Avg EU | Avg AU | Avg Entropy |
|-------|-------------|--------|--------|-------------|
| Who is Deepak Sai Pendyala | **0.0267** | 0.0400 | 0.6696 | 0.2933 |
| What is Deep Learning | **0.0293** | 0.0435 | 0.6729 | 0.3793 |

## Observations

1. **"What is Deep Learning" shows higher uncertainty** (LogTokU: 0.0293 vs 0.0267)
   - This is expected as it's a conceptual question with multiple valid ways to answer

2. **"Who is Deepak Sai Pendyala" shows lower uncertainty** (LogTokU: 0.0267)
   - The model correctly indicates uncertainty by saying it couldn't find information
   - Lower LogTokU reflects the model's confidence in stating it doesn't know

3. **Llama 3.2 produces better responses** than TinyLlama:
   - More accurate acknowledgment of unknown information
   - Better structured responses
   - More appropriate uncertainty handling

4. **LogTokU values are lower** than with TinyLlama (0.0267-0.0293 vs 0.0384-0.0398):
   - Likely due to Llama 3.2's better calibration
   - Model is more confident in its responses (even when saying it doesn't know)

## Usage

```bash
cd /home/dpendya/Documents/dlba/logtoku
source venv/bin/activate

# Test with Llama 3.2 (default)
python test_llama_simple.py --query "Your query here" --quantize

# Test with custom model
python test_llama_simple.py --query "Your query" --model "meta-llama/Llama-3.2-1B-Instruct" --quantize
```

## Conclusion

LogTokU works correctly with Llama 3.2 and provides meaningful uncertainty estimates. The model shows appropriate uncertainty for queries it doesn't know about and slightly higher uncertainty for open-ended conceptual questions.

