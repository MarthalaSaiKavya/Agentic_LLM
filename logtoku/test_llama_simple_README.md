# Simple LogTokU Test Script for Llama Models

A clean, minimal script to test LogTokU uncertainty estimation with Llama models.

## Usage

```bash
cd /home/dpendya/Documents/dlba/logtoku
source venv/bin/activate

# Basic usage
python test_llama_simple.py --query "Your query here"

# With custom model
python test_llama_simple.py --query "Your query here" --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# With quantization (recommended for larger models)
python test_llama_simple.py --query "Your query here" --quantize

# Custom number of tokens
python test_llama_simple.py --query "Your query here" --max_tokens 50
```

## Examples

```bash
# Test with default model (TinyLlama)
python test_llama_simple.py --query "Who is Deepak Sai Pendyala" --quantize

# Test with Meta Llama (requires access)
python test_llama_simple.py --query "What is Deep Learning" \
                            --model "meta-llama/Llama-2-7b-chat-hf" \
                            --quantize
```

## Output

The script outputs:
- Generated response
- Average EU (Epistemic Uncertainty)
- Average AU (Aleatoric Uncertainty)
- Average LogTokU (EU × AU) - main uncertainty metric
- Average Entropy
- Maximum LogTokU
- Number of tokens analyzed

## Notes

- Higher LogTokU = Higher uncertainty
- Use `--quantize` for models larger than 3B to save memory
- For Meta Llama models, request access at https://huggingface.co/meta-llama/
- Login to Hugging Face: `huggingface-cli login`

