"""
Test LogTokU with Llama models (as used in the original paper).
This script tests both the original and fixed implementations with Llama models.
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.special import softmax, digamma
import argparse
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SenU'))
from metrics import get_eu, topk

warnings.filterwarnings("ignore")

def calculate_entropy(logits):
    """Calculate entropy from logits"""
    probs = softmax(logits)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy

def calculate_logtoku_original(logits, k=2):
    """
    Original LogTokU implementation (as in the paper)
    """
    eu_func = get_eu(mode="eu", k=k)
    au_func = get_eu(mode="au", k=k)
    
    try:
        eu = eu_func(logits)
        au = au_func(logits)
        
        # Handle NaN
        if np.isnan(au) or au <= 0:
            return None
        
        logtoku = eu * au
        entropy = calculate_entropy(logits)
        
        return {
            'EU': eu,
            'AU': au,
            'LogTokU': logtoku,
            'Entropy': entropy,
            'method': 'original'
        }
    except Exception as e:
        return None

def calculate_logtoku_fixed(logits, k=2):
    """
    Fixed LogTokU implementation (handles negative logits)
    """
    top_k = k
    if len(logits) < top_k:
        raise ValueError("Logits array length is less than top_k.")
    
    top_values = np.partition(logits, -top_k)[-top_k:]
    
    # EU calculation - fixed for negative logits
    top_values_shifted = top_values - top_values.max()
    exp_top = np.exp(top_values_shifted)
    eu = top_k / (np.sum(exp_top) + top_k)
    
    # AU calculation - shift to positive
    top_values_for_au = top_values - np.min(top_values) + 1.0
    
    try:
        alpha = np.array([top_values_for_au])
        alpha_0 = alpha.sum(axis=1, keepdims=True)
        psi_alpha_k_plus_1 = digamma(alpha + 1)
        psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
        result = - (alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
        au = result.sum(axis=1)[0]
        
        if np.isnan(au) or au <= 0:
            au = np.std(top_values_for_au) / (np.mean(top_values_for_au) + 1e-10)
    except:
        au = np.std(top_values) / (abs(np.mean(top_values)) + 1e-10)
    
    logtoku = eu * au
    entropy = calculate_entropy(logits)
    
    return {
        'EU': eu,
        'AU': au,
        'LogTokU': logtoku,
        'Entropy': entropy,
        'method': 'fixed'
    }

def analyze_query(model, tokenizer, query, device, max_new_tokens=50, use_quantization=False):
    """Analyze query with both original and fixed implementations"""
    model.eval()
    
    # Format query appropriately for the model
    if 'llama' in model.config.name_or_path.lower():
        if 'llama2' in model.config.name_or_path.lower():
            # Llama-2 format
            formatted_query = f"<s>[INST] {query} [/INST]"
        else:
            # Llama-3 format
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            formatted_query = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    else:
        formatted_query = query
    
    inputs = tokenizer(formatted_query, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
        )
    
    generated_ids = outputs.sequences[0]
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    scores = outputs.scores
    token_metrics_original = []
    token_metrics_fixed = []
    generated_text_tokens = []
    input_length = input_ids.shape[1]
    
    for i, score_tensor in enumerate(scores):
        logits = score_tensor[0].cpu().numpy()
        predicted_token_id = torch.argmax(score_tensor[0]).item()
        predicted_token = tokenizer.decode([predicted_token_id], skip_special_tokens=False)
        
        # Try original implementation
        metrics_orig = calculate_logtoku_original(logits, k=2)
        if metrics_orig:
            metrics_orig['token'] = predicted_token
            metrics_orig['position'] = input_length + i
            token_metrics_original.append(metrics_orig)
        
        # Try fixed implementation
        try:
            metrics_fixed = calculate_logtoku_fixed(logits, k=2)
            metrics_fixed['token'] = predicted_token
            metrics_fixed['position'] = input_length + i
            token_metrics_fixed.append(metrics_fixed)
        except:
            pass
        
        generated_text_tokens.append(predicted_token)
    
    # Calculate aggregates
    def calc_aggregates(metrics_list):
        if not metrics_list:
            return None
        eus = [m['EU'] for m in metrics_list]
        aus = [m['AU'] for m in metrics_list]
        logtokus = [m['LogTokU'] for m in metrics_list]
        entropies = [m['Entropy'] for m in metrics_list]
        
        return {
            'avg_EU': np.mean(eus),
            'avg_AU': np.mean(aus),
            'avg_LogTokU': np.mean(logtokus),
            'avg_Entropy': np.mean(entropies),
            'max_LogTokU': max(logtokus),
            'num_tokens': len(metrics_list)
        }
    
    return {
        'query': query,
        'response': full_response,
        'original_metrics': calc_aggregates(token_metrics_original),
        'fixed_metrics': calc_aggregates(token_metrics_fixed),
        'token_metrics_original': token_metrics_original[:10],  # First 10 for display
        'token_metrics_fixed': token_metrics_fixed[:10]
    }

def print_results(results):
    """Print formatted results"""
    print("\n" + "="*80)
    print(f"Query: {results['query']}")
    print("="*80)
    print(f"\nGenerated Response:")
    print(f"{results['response'][:200]}...")  # First 200 chars
    print("\n" + "-"*80)
    
    if results['original_metrics']:
        orig = results['original_metrics']
        print(f"\nOriginal Implementation Results:")
        print(f"  Average EU:      {orig['avg_EU']:.4f}")
        print(f"  Average AU:      {orig['avg_AU']:.4f}")
        print(f"  Average LogTokU: {orig['avg_LogTokU']:.4f}")
        print(f"  Average Entropy: {orig['avg_Entropy']:.4f}")
        print(f"  Max LogTokU:     {orig['max_LogTokU']:.4f}")
        print(f"  Tokens analyzed: {orig['num_tokens']}")
    
    if results['fixed_metrics']:
        fixed = results['fixed_metrics']
        print(f"\nFixed Implementation Results:")
        print(f"  Average EU:      {fixed['avg_EU']:.4f}")
        print(f"  Average AU:      {fixed['avg_AU']:.4f}")
        print(f"  Average LogTokU: {fixed['avg_LogTokU']:.4f}")
        print(f"  Average Entropy: {fixed['avg_Entropy']:.4f}")
        print(f"  Max LogTokU:     {fixed['max_LogTokU']:.4f}")
        print(f"  Tokens analyzed: {fixed['num_tokens']}")

def load_llama_model(model_name, device, use_quantization=False):
    """Load Llama model with optional quantization"""
    print(f"\nLoading model: {model_name}")
    print(f"Device: {device}")
    print(f"Quantization: {use_quantization}")
    
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
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Llama model name (default: meta-llama/Llama-3.2-3B-Instruct)"
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
        default=40,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 4-bit quantization (recommended for larger models)"
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
        model, tokenizer = load_llama_model(args.model, device, args.quantize)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nNote: You may need to:")
        print("  1. Configure Hugging Face token: huggingface-cli login")
        print("  2. Request access to Llama models on Hugging Face")
        print("  3. Try a smaller model or use quantization")
        return
    
    # Test queries
    all_results = []
    for query in args.queries:
        try:
            print(f"\n{'='*80}")
            print(f"Processing: {query}")
            print(f"{'='*80}")
            results = analyze_query(model, tokenizer, query, device, args.max_tokens, args.quantize)
            all_results.append(results)
            print_results(results)
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        if all_results[0]['original_metrics']:
            print("\nOriginal Implementation:")
            print(f"{'Query':<40} {'Avg LogTokU':<15} {'Avg EU':<12} {'Avg AU':<12}")
            print("-"*80)
            for results in all_results:
                if results['original_metrics']:
                    orig = results['original_metrics']
                    query_short = results['query'][:38]
                    print(f"{query_short:<40} {orig['avg_LogTokU']:<15.4f} "
                          f"{orig['avg_EU']:<12.4f} {orig['avg_AU']:<12.4f}")
        
        if all_results[0]['fixed_metrics']:
            print("\nFixed Implementation:")
            print(f"{'Query':<40} {'Avg LogTokU':<15} {'Avg EU':<12} {'Avg AU':<12}")
            print("-"*80)
            for results in all_results:
                if results['fixed_metrics']:
                    fixed = results['fixed_metrics']
                    query_short = results['query'][:38]
                    print(f"{query_short:<40} {fixed['avg_LogTokU']:<15.4f} "
                          f"{fixed['avg_EU']:<12.4f} {fixed['avg_AU']:<12.4f}")

if __name__ == "__main__":
    main()

