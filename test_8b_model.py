"""Simple test to verify Llama-3.1 8B model setup."""

import os
from huggingface_hub import login
from src.token_self_repair.llm import load_llama

# Ensure token is available
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)

print("Loading Llama-3.1 8B Instruct model...")
print("(First time will download ~16GB; ensure adequate GPU/CPU and disk.)")

# Try to load with quantization (saves memory)
try:
    provider = load_llama("meta-llama/Llama-3.1-8B-Instruct", quantize=True)
    print("✅ Model loaded successfully!")
    
    # Test generation
    print("\nTesting generation...")
    # Use generate_with_logits to get text directly
    tokens, logits = provider.generate_with_logits("Say hello in one sentence.", max_new_tokens=20)
    response = provider.tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"Response: {response}")
    
    print("\n✅ Everything works! You can now use the 8B model.")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Verify: huggingface-cli login (run again if needed)")
    print("3. Verify license accepted at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("4. Check firewall/proxy settings")
    print("5. Try: python -c \"from huggingface_hub import model_info; print(model_info('meta-llama/Llama-3.1-8B-Instruct'))\"")


