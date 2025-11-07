"""
Test LogTokU uncertainty estimation for specific queries.
This script evaluates uncertainty for given questions using LogTokU metrics.
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.special import softmax, digamma
import argparse
from tqdm import tqdm

# Add SenU to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SenU'))
from metrics import get_eu, topk

def calculate_entropy(logits):
    """Calculate entropy from logits"""
    probs = softmax(logits)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy

def calculate_logtoku_per_token(logits, k=2):
    """
    Calculate LogTokU (EU * AU) for each token's logits.
    
    Args:
        logits: numpy array of logits for a single token
        k: top-k value for EU and AU calculation
        
    Returns:
        dict with EU, AU, LogTokU, and entropy values
    """
    eu_func = get_eu(mode="eu", k=k)
    au_func = get_eu(mode="au", k=k)
    
    # Shift logits to make them positive for AU calculation
    # This is necessary because AU uses digamma which requires positive values
    # We subtract the minimum to make all values >= 0, then add a small epsilon
    logits_shifted = logits - np.min(logits) + 1e-10
    
    eu = eu_func(logits)
    try:
        au = au_func(logits_shifted)
        # Handle NaN or negative AU values
        if np.isnan(au) or au <= 0:
            au = 1e-10  # Small positive value as fallback
    except:
        au = 1e-10
    
    logtoku = eu * au
    entropy = calculate_entropy(logits)
    
    return {
        'EU': eu,
        'AU': au,
        'LogTokU': logtoku,
        'Entropy': entropy
    }

def analyze_query(model, tokenizer, query, device, max_new_tokens=50):
    """
    Analyze a query and calculate LogTokU uncertainty for each generated token.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        query: The input query string
        device: Device to run on
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        dict with results
    """
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(query, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Generate with output_scores to get logits
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the full response
    generated_ids = outputs.sequences[0]
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Get logits for each generated token
    scores = outputs.scores  # List of tensors, one per generated token
    
    # Calculate metrics for each token
    token_metrics = []
    generated_text_tokens = []
    
    input_length = input_ids.shape[1]
    
    for i, score_tensor in enumerate(scores):
        # Get logits for this token (shape: [1, vocab_size])
        logits = score_tensor[0].cpu().numpy()
        
        # Get the predicted token
        predicted_token_id = torch.argmax(score_tensor[0]).item()
        predicted_token = tokenizer.decode([predicted_token_id], skip_special_tokens=False)
        
        # Calculate metrics
        metrics = calculate_logtoku_per_token(logits, k=2)
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
    
    # Average uncertainty (higher = more uncertain)
    avg_eu = np.mean(eus) if eus else 0
    avg_au = np.mean(aus) if aus else 0
    avg_logtoku = np.mean(logtokus) if logtokus else 0
    avg_entropy = np.mean(entropies) if entropies else 0
    
    # Max uncertainty (peak uncertainty)
    max_logtoku = max(logtokus) if logtokus else 0
    max_eu = max(eus) if eus else 0
    max_au = max(aus) if aus else 0
    
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
            'max_EU': max_eu,
            'max_AU': max_au,
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
    print(f"\nAggregate Uncertainty Metrics:")
    print(f"  Average EU (Epistemic Uncertainty):  {agg['avg_EU']:.4f}")
    print(f"  Average AU (Aleatoric Uncertainty):  {agg['avg_AU']:.4f}")
    print(f"  Average LogTokU (EU × AU):           {agg['avg_LogTokU']:.4f}")
    print(f"  Average Entropy:                     {agg['avg_Entropy']:.4f}")
    print(f"  Maximum LogTokU:                     {agg['max_LogTokU']:.4f}")
    print(f"  Number of Generated Tokens:          {agg['num_tokens']}")
    
    print(f"\nToken-by-Token Uncertainty (first 10 tokens):")
    print(f"{'Pos':<6} {'Token':<20} {'EU':<10} {'AU':<10} {'LogTokU':<12} {'Entropy':<10}")
    print("-"*80)
    
    for i, metrics in enumerate(results['token_metrics'][:10]):
        token_display = metrics['token'].replace('\n', '\\n')[:18]
        print(f"{metrics['position']:<6} {token_display:<20} "
              f"{metrics['EU']:<10.4f} {metrics['AU']:<10.4f} "
              f"{metrics['LogTokU']:<12.4f} {metrics['Entropy']:<10.4f}")
    
    if len(results['token_metrics']) > 10:
        print(f"... ({len(results['token_metrics']) - 10} more tokens)")

def load_model(model_name, device, use_quantization=False):
    """Load model and tokenizer"""
    print(f"\nLoading model: {model_name}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if use_quantization and device.type == 'cuda':
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map="auto" if device.type == 'cuda' else None
        )
        if device.type == 'cpu':
            model = model.to(device)
    
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Test LogTokU on specific queries")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path (default: gpt2)"
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=["Who is Deepak Sai Pendyala", "What is Deep Learning"],
        help="Queries to test"
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
        help="Use 4-bit quantization (CUDA only)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU")
    else:
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    
    # Load model
    try:
        model, tokenizer = load_model(args.model, device, args.quantize)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying with a smaller default model...")
        model, tokenizer = load_model("gpt2", device, False)
    
    # Test each query
    all_results = []
    for query in args.queries:
        try:
            print(f"\n{'='*80}")
            print(f"Processing query: {query}")
            print(f"{'='*80}")
            results = analyze_query(model, tokenizer, query, device, args.max_tokens)
            all_results.append(results)
            print_results(results)
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison summary
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Query':<40} {'Avg LogTokU':<15} {'Avg EU':<12} {'Avg AU':<12} {'Avg Entropy':<15}")
        print("-"*80)
        for results in all_results:
            agg = results['aggregate_metrics']
            query_short = results['query'][:38]
            print(f"{query_short:<40} {agg['avg_LogTokU']:<15.4f} "
                  f"{agg['avg_EU']:<12.4f} {agg['avg_AU']:<12.4f} "
                  f"{agg['avg_Entropy']:<15.4f}")
        
        # Find which query has higher uncertainty
        uncertainties = [(r['aggregate_metrics']['avg_LogTokU'], r['query']) 
                        for r in all_results]
        uncertainties.sort(reverse=True)
        print(f"\n{'='*80}")
        print("INSIGHTS:")
        print(f"{'='*80}")
        print(f"Higher uncertainty query: {uncertainties[0][1]}")
        print(f"  - Average LogTokU: {uncertainties[0][0]:.4f}")
        print(f"  - Interpretation: Model shows higher uncertainty in generating response")
        print(f"\nLower uncertainty query: {uncertainties[-1][1]}")
        print(f"  - Average LogTokU: {uncertainties[-1][0]:.4f}")
        print(f"  - Interpretation: Model shows lower uncertainty in generating response")
        print(f"\nNote: Higher LogTokU indicates higher uncertainty. However, note that")
        print(f"      LLMs can be confidently wrong, so high certainty doesn't always")
        print(f"      mean correctness, especially for factual queries about specific people.")

if __name__ == "__main__":
    main()

