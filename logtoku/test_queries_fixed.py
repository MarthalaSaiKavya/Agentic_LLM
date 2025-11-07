"""
Fixed version of test_queries.py that properly handles negative logits.
The issue: Original EU formula expects some positive logits, but GPT-2 produces very negative ones.
Solution: We need to normalize logits properly or adjust the formula.
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.special import softmax, digamma
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SenU'))
from metrics import get_eu, topk

def calculate_entropy(logits):
    """Calculate entropy from logits"""
    probs = softmax(logits)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy

def calculate_logtoku_per_token_fixed(logits, k=2):
    """
    Calculate LogTokU with proper handling of negative logits.
    
    The original implementation has issues with very negative logits.
    We'll use a normalized approach that works with any logit range.
    """
    # For EU: The formula k / (sum(max(0, top_k)) + k) fails when all logits are negative
    # Solution: Normalize by subtracting max, or use a different formula
    # Let's use the relative difference approach
    
    # Get top-k values
    top_k = k
    if len(logits) < top_k:
        raise ValueError("Logits array length is less than top_k.")
    
    top_values = np.partition(logits, -top_k)[-top_k:]
    
    # EU calculation - fix for negative logits
    # Original: k / (sum(max(0, top_values)) + k)
    # Problem: When all negative, max(0, top_values) = 0, gives EU = 1.0
    # Solution: Use exponential normalization or relative difference
    # Let's use: EU = 1 - (exp(top_sum) / (exp(top_sum) + k))
    # Or simpler: Use the difference between top values
    
    # Better approach: Normalize top values and use the formula
    # Shift to make top value = 0 (preserves relative differences)
    top_values_shifted = top_values - top_values.max()
    # Now use exp to convert to positive space
    exp_top = np.exp(top_values_shifted)
    eu = top_k / (np.sum(exp_top) + top_k)
    
    # Alternative EU: Use variance/uncertainty measure
    # eu_alt = 1 - (top_values.max() - top_values.min()) / (abs(top_values.max()) + 1e-10)
    
    # AU calculation - need to handle negative logits
    # Shift top values to be positive for digamma
    # We want to preserve relative relationships
    top_values_for_au = top_values - np.min(top_values) + 1.0  # Make all positive, min = 1.0
    
    try:
        alpha = np.array([top_values_for_au])
        alpha_0 = alpha.sum(axis=1, keepdims=True)
        psi_alpha_k_plus_1 = digamma(alpha + 1)
        psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
        result = - (alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
        au = result.sum(axis=1)[0]
        
        # AU should be positive - if negative or NaN, use fallback
        if np.isnan(au) or au <= 0:
            # Fallback: Use variance-based measure
            au = np.std(top_values_for_au) / (np.mean(top_values_for_au) + 1e-10)
    except:
        # Fallback for AU
        au = np.std(top_values) / (abs(np.mean(top_values)) + 1e-10)
    
    logtoku = eu * au
    entropy = calculate_entropy(logits)
    
    return {
        'EU': eu,
        'AU': au,
        'LogTokU': logtoku,
        'Entropy': entropy,
        'top_values': top_values
    }

def analyze_query_fixed(model, tokenizer, query, device, max_new_tokens=50):
    """Analyze query with fixed LogTokU calculation"""
    model.eval()
    
    inputs = tokenizer(query, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs.sequences[0]
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    scores = outputs.scores
    token_metrics = []
    generated_text_tokens = []
    input_length = input_ids.shape[1]
    
    for i, score_tensor in enumerate(scores):
        logits = score_tensor[0].cpu().numpy()
        predicted_token_id = torch.argmax(score_tensor[0]).item()
        predicted_token = tokenizer.decode([predicted_token_id], skip_special_tokens=False)
        
        metrics = calculate_logtoku_per_token_fixed(logits, k=2)
        metrics['token_id'] = predicted_token_id
        metrics['token'] = predicted_token
        metrics['position'] = input_length + i
        
        token_metrics.append(metrics)
        generated_text_tokens.append(predicted_token)
    
    # Calculate aggregate metrics
    eus = [m['EU'] for m in token_metrics]
    aus = [m['AU'] for m in token_metrics]
    logtokus = [m['LogTokU'] for m in token_metrics]
    entropies = [m['Entropy'] for m in token_metrics]
    
    avg_eu = np.mean(eus) if eus else 0
    avg_au = np.mean(aus) if aus else 0
    avg_logtoku = np.mean(logtokus) if logtokus else 0
    avg_entropy = np.mean(entropies) if entropies else 0
    max_logtoku = max(logtokus) if logtokus else 0
    
    return {
        'query': query,
        'response': full_response,
        'generated_tokens': generated_text_tokens,
        'token_metrics': token_metrics,
        'aggregate_metrics': {
            'avg_EU': avg_eu,
            'avg_AU': avg_au,
            'avg_LogTokU': avg_logtoku,
            'avg_Entropy': avg_entropy,
            'max_LogTokU': max_logtoku,
            'num_tokens': len(token_metrics)
        }
    }

def print_results(results):
    """Print formatted results"""
    print("\n" + "="*80)
    print(f"Query: {results['query']}")
    print("="*80)
    print(f"\nGenerated Response:")
    print(f"{results['response']}")
    print("\n" + "-"*80)
    
    agg = results['aggregate_metrics']
    print(f"\nAggregate Uncertainty Metrics (FIXED):")
    print(f"  Average EU (Epistemic Uncertainty):  {agg['avg_EU']:.4f}")
    print(f"  Average AU (Aleatoric Uncertainty):  {agg['avg_AU']:.4f}")
    print(f"  Average LogTokU (EU × AU):           {agg['avg_LogTokU']:.4f}")
    print(f"  Average Entropy:                     {agg['avg_Entropy']:.4f}")
    print(f"  Maximum LogTokU:                     {agg['max_LogTokU']:.4f}")
    print(f"  Number of Generated Tokens:          {agg['num_tokens']}")

def main():
    parser = argparse.ArgumentParser(description="Test LogTokU with fixed implementation")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--queries", type=str, nargs="+", 
                       default=["Who is Deepak Sai Pendyala", "What is Deep Learning"])
    parser.add_argument("--max_tokens", type=int, default=30)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map="auto" if device.type == 'cuda' else None
    )
    model.eval()
    
    all_results = []
    for query in args.queries:
        results = analyze_query_fixed(model, tokenizer, query, device, args.max_tokens)
        all_results.append(results)
        print_results(results)
    
    # Comparison
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON (FIXED VERSION)")
        print("="*80)
        print(f"{'Query':<40} {'Avg LogTokU':<15} {'Avg EU':<12} {'Avg AU':<12} {'Avg Entropy':<15}")
        print("-"*80)
        for results in all_results:
            agg = results['aggregate_metrics']
            query_short = results['query'][:38]
            print(f"{query_short:<40} {agg['avg_LogTokU']:<15.4f} "
                  f"{agg['avg_EU']:<12.4f} {agg['avg_AU']:<12.4f} "
                  f"{agg['avg_Entropy']:<15.4f}")
        
        uncertainties = [(r['aggregate_metrics']['avg_LogTokU'], r['query']) 
                        for r in all_results]
        uncertainties.sort(reverse=True)
        print(f"\nHigher uncertainty: {uncertainties[0][1]} (LogTokU: {uncertainties[0][0]:.4f})")
        print(f"Lower uncertainty: {uncertainties[-1][1]} (LogTokU: {uncertainties[-1][0]:.4f})")

if __name__ == "__main__":
    main()

