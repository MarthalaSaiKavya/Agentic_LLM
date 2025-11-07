"""
Simple test script for LogTokU with Llama models.
Usage: python test_llama_simple.py --query "Your query here"
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.special import softmax
import argparse
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SenU'))
from metrics import get_eu

warnings.filterwarnings("ignore")

def calculate_entropy(logits):
    """Calculate entropy from logits"""
    probs = softmax(logits)
    return -np.sum(probs * np.log(probs + 1e-10))

def calculate_logtoku(logits, k=2):
    """Calculate LogTokU metrics"""
    eu_func = get_eu(mode="eu", k=k)
    au_func = get_eu(mode="au", k=k)
    
    eu = eu_func(logits)
    au = au_func(logits)
    
    if np.isnan(au) or au <= 0:
        return None
    
    logtoku = eu * au
    entropy = calculate_entropy(logits)
    
    return {
        'EU': eu,
        'AU': au,
        'LogTokU': logtoku,
        'Entropy': entropy
    }

def analyze_query(model, tokenizer, query, device, max_new_tokens=50):
    """Analyze query and calculate LogTokU for each token"""
    model.eval()
    
    # Format query for Llama models
    if 'llama2' in model.config.name_or_path.lower():
        formatted_query = f"<s>[INST] {query} [/INST]"
    elif 'llama' in model.config.name_or_path.lower():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        formatted_query = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_query = query
    
    inputs = tokenizer(formatted_query, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Get pad_token_id
    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.unk_token_id
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=pad_token_id
        )
    
    # Decode only the generated tokens (excluding input)
    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    scores = outputs.scores
    
    token_metrics = []
    for i, score_tensor in enumerate(scores):
        logits = score_tensor[0].cpu().numpy()
        metrics = calculate_logtoku(logits, k=2)
        if metrics:
            token_metrics.append(metrics)
    
    if not token_metrics:
        return None
    
    # Calculate aggregates
    eus = [m['EU'] for m in token_metrics]
    aus = [m['AU'] for m in token_metrics]
    logtokus = [m['LogTokU'] for m in token_metrics]
    entropies = [m['Entropy'] for m in token_metrics]
    
    return {
        'query': query,
        'response': response,
        'avg_EU': np.mean(eus),
        'avg_AU': np.mean(aus),
        'avg_LogTokU': np.mean(logtokus),
        'avg_Entropy': np.mean(entropies),
        'max_LogTokU': max(logtokus),
        'num_tokens': len(token_metrics)
    }

def load_model(model_name, device, use_quantization=False):
    """Load Llama model"""
    print(f"Loading model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_quantization and device.type == 'cuda':
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map="auto" if device.type == 'cuda' else None,
            trust_remote_code=True
        )
        if device.type == 'cpu':
            model = model.to(device)
    
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Test LogTokU with Llama models")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to test"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name (default: meta-llama/Llama-3.2-3B-Instruct)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 4-bit quantization"
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    try:
        model, tokenizer = load_model(args.model, device, args.quantize)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nNote: For Meta Llama models, you need to:")
        print("  1. Request access at https://huggingface.co/meta-llama/")
        print("  2. Login: huggingface-cli login")
        return
    
    print(f"\nQuery: {args.query}")
    print("="*60)
    
    results = analyze_query(model, tokenizer, args.query, device, args.max_tokens)
    
    if results:
        print(f"\nResponse: {results['response'][:200]}...")
        print("\n" + "-"*60)
        print("LogTokU Uncertainty Metrics:")
        print(f"  Average EU (Epistemic Uncertainty):  {results['avg_EU']:.4f}")
        print(f"  Average AU (Aleatoric Uncertainty):  {results['avg_AU']:.4f}")
        print(f"  Average LogTokU (EU × AU):           {results['avg_LogTokU']:.4f}")
        print(f"  Average Entropy:                     {results['avg_Entropy']:.4f}")
        print(f"  Maximum LogTokU:                     {results['max_LogTokU']:.4f}")
        print(f"  Tokens analyzed:                     {results['num_tokens']}")
        print("\nNote: Higher LogTokU indicates higher uncertainty")
    else:
        print("Error: Could not calculate metrics")

if __name__ == "__main__":
    main()

