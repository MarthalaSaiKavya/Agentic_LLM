"""Simple test to verify Llama-3 8B model setup."""

from src.token_self_repair.llm import load_llama

print("Loading Llama-3 8B Instruct model...")
print("(First time will download ~16GB, this may take a while)")

# Try to load with quantization (saves memory)
try:
    provider = load_llama("meta-llama/Llama-3.1-8B-Instruct", quantize=True)
    print("✅ Model loaded successfully!")
    
    # Test generation
    print("\nTesting generation...")
    response = provider.generate("Say hello in one sentence.")
    print(f"Response: {response}")
    
    print("\n✅ Everything works! You can now use the 8B model.")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nMake sure you:")
    print("1. Ran: huggingface-cli login")
    print("2. Accepted the license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("3. Have enough GPU memory (or use CPU)")


