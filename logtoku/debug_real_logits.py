"""
Debug script to examine actual logits from model generation
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.special import softmax, digamma
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SenU'))
from metrics import get_eu, topk

def examine_logits(query, model, tokenizer, device, num_tokens=5):
    """Examine actual logits from a query"""
    model.eval()
    
    inputs = tokenizer(query, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=num_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    scores = outputs.scores
    
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    for i, score_tensor in enumerate(scores):
        logits = score_tensor[0].cpu().numpy()
        
        # Get top-5 logits
        top_5_indices = np.argsort(logits)[-5:][::-1]
        top_5_values = logits[top_5_indices]
        top_5_tokens = [tokenizer.decode([idx]) for idx in top_5_indices]
        
        # Calculate metrics
        eu_func = get_eu(mode="eu", k=2)
        au_func = get_eu(mode="au", k=2)
        
        eu = eu_func(logits)
        try:
            au = au_func(logits)  # Use original without shifting
            logtoku = eu * au
        except Exception as e:
            print(f"  ERROR calculating AU: {e}")
            # Try with shifted
            logits_shifted = logits - np.min(logits) + 1.0
            au = au_func(logits_shifted)
            logtoku = eu * au
        
        entropy = -np.sum(softmax(logits) * np.log(softmax(logits) + 1e-10))
        
        predicted_token_id = np.argmax(logits)
        predicted_token = tokenizer.decode([predicted_token_id])
        
        print(f"\nToken {i+1}: '{predicted_token}' (ID: {predicted_token_id})")
        print(f"  Top-5 logits:")
        for j, (val, token) in enumerate(zip(top_5_values, top_5_tokens)):
            print(f"    {j+1}. {token:15s} {val:8.2f}")
        
        print(f"  Metrics:")
        print(f"    EU:       {eu:.6f}")
        print(f"    AU:       {au:.6f}")
        print(f"    LogTokU:  {logtoku:.6f}")
        print(f"    Entropy:  {entropy:.6f}")
        print(f"  Top-2 logits: {np.partition(logits, -2)[-2:]}")
        print(f"  Min logit: {np.min(logits):.2f}, Max logit: {np.max(logits):.2f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "gpt2"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map="auto" if device.type == 'cuda' else None
    )
    model.eval()
    
    queries = [
        "Who is Deepak Sai Pendyala",
        "What is Deep Learning"
    ]
    
    for query in queries:
        examine_logits(query, model, tokenizer, device, num_tokens=5)

if __name__ == "__main__":
    main()

