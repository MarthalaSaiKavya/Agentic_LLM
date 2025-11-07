"""
Diagnostic script to check if LogTokU is working correctly.
This will test with known high/low uncertainty cases.
"""

import numpy as np
import sys
import os
from scipy.special import softmax, digamma

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SenU'))
from metrics import get_eu, topk

def calculate_entropy(logits):
    """Calculate entropy from logits"""
    probs = softmax(logits)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy

def test_au_calculation():
    """Test AU calculation with different logit scenarios"""
    print("="*80)
    print("Testing AU Calculation")
    print("="*80)
    
    # Test case 1: Very confident (one high logit, others very low)
    confident_logits = np.array([10.0, 0.1, 0.05, -2.0, -3.0, -4.0, -5.0])
    
    # Test case 2: Uncertain (multiple similar logits)
    uncertain_logits = np.array([2.0, 1.9, 1.8, 1.7, -2.0, -3.0, -4.0])
    
    # Test case 3: Very uncertain (all similar)
    very_uncertain_logits = np.array([1.0, 0.9, 0.8, 0.7, 0.6, -1.0, -2.0])
    
    au_func = get_eu(mode="au", k=2)
    
    print("\nTest Case 1: Very Confident (one high logit)")
    print(f"Logits: {confident_logits[:5]}...")
    try:
        au1 = au_func(confident_logits)
        print(f"AU: {au1:.4f}")
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"  Min logit: {np.min(confident_logits)}")
        print(f"  Max logit: {np.max(confident_logits)}")
        top_2 = np.partition(confident_logits, -2)[-2:]
        print(f"  Top-2 logits: {top_2}")
        print(f"  Top-2 after +1: {top_2 + 1}")
    
    print("\nTest Case 2: Uncertain (multiple similar logits)")
    print(f"Logits: {uncertain_logits[:5]}...")
    try:
        au2 = au_func(uncertain_logits)
        print(f"AU: {au2:.4f}")
    except Exception as e:
        print(f"ERROR: {e}")
        top_2 = np.partition(uncertain_logits, -2)[-2:]
        print(f"  Top-2 logits: {top_2}")
    
    print("\nTest Case 3: Very Uncertain")
    print(f"Logits: {very_uncertain_logits[:5]}...")
    try:
        au3 = au_func(very_uncertain_logits)
        print(f"AU: {au3:.4f}")
    except Exception as e:
        print(f"ERROR: {e}")
        top_2 = np.partition(very_uncertain_logits, -2)[-2:]
        print(f"  Top-2 logits: {top_2}")
    
    print("\n" + "="*80)
    print("Expected: Higher uncertainty should have higher AU")
    print("="*80)

def test_eu_calculation():
    """Test EU calculation"""
    print("\n" + "="*80)
    print("Testing EU Calculation")
    print("="*80)
    
    confident_logits = np.array([10.0, 0.1, 0.05, -2.0, -3.0, -4.0, -5.0])
    uncertain_logits = np.array([2.0, 1.9, 1.8, 1.7, -2.0, -3.0, -4.0])
    
    eu_func = get_eu(mode="eu", k=2)
    
    eu1 = eu_func(confident_logits)
    eu2 = eu_func(uncertain_logits)
    
    print(f"\nConfident logits EU: {eu1:.4f}")
    print(f"Uncertain logits EU: {eu2:.4f}")
    print(f"\nExpected: Higher uncertainty should have higher EU")
    print(f"Result: {'PASS' if eu2 > eu1 else 'FAIL'}")

def test_with_real_model_logits():
    """Test with actual logits from a model generation"""
    print("\n" + "="*80)
    print("Testing with Real Model-like Logits")
    print("="*80)
    
    # Simulate realistic logits from GPT-2 (vocab size 50257)
    # Create a sparse array with mostly negative values
    vocab_size = 50257
    
    # High confidence case: one token has very high logit
    confident_logits = np.full(vocab_size, -10.0)
    confident_logits[1000] = 15.0  # One very confident token
    confident_logits[2000] = 2.0   # Second choice much lower
    
    # Uncertain case: top tokens have similar logits
    uncertain_logits = np.full(vocab_size, -10.0)
    uncertain_logits[1000] = 5.0
    uncertain_logits[2000] = 4.8
    uncertain_logits[3000] = 4.6
    uncertain_logits[4000] = 4.4
    
    eu_func = get_eu(mode="eu", k=2)
    au_func = get_eu(mode="au", k=2)
    
    print("\nHigh Confidence Case:")
    eu_conf = eu_func(confident_logits)
    try:
        au_conf = au_func(confident_logits)
        logtoku_conf = eu_conf * au_conf
        print(f"  EU: {eu_conf:.4f}")
        print(f"  AU: {au_conf:.4f}")
        print(f"  LogTokU: {logtoku_conf:.4f}")
    except Exception as e:
        print(f"  EU: {eu_conf:.4f}")
        print(f"  AU: ERROR - {e}")
        top_2 = np.partition(confident_logits, -2)[-2:]
        print(f"  Top-2 logits: {top_2}")
        print(f"  Issue: Negative logits in digamma calculation")
    
    print("\nUncertain Case:")
    eu_unc = eu_func(uncertain_logits)
    try:
        au_unc = au_func(uncertain_logits)
        logtoku_unc = eu_unc * au_unc
        print(f"  EU: {eu_unc:.4f}")
        print(f"  AU: {au_unc:.4f}")
        print(f"  LogTokU: {logtoku_unc:.4f}")
    except Exception as e:
        print(f"  EU: {eu_unc:.4f}")
        print(f"  AU: ERROR - {e}")
        top_2 = np.partition(uncertain_logits, -2)[-2:]
        print(f"  Top-2 logits: {top_2}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    print("The AU calculation uses digamma(alpha + 1) where alpha are the logits.")
    print("Digamma requires positive inputs, but logits can be negative.")
    print("This is likely why we're seeing issues.")
    print("\nPossible solutions:")
    print("1. Shift logits to be positive (current approach)")
    print("2. Use exp(logits) to convert to positive values")
    print("3. Use a different normalization approach")

def test_logit_shift_approach():
    """Test different approaches to handle negative logits"""
    print("\n" + "="*80)
    print("Testing Different Logit Handling Approaches")
    print("="*80)
    
    # Realistic uncertain logits (top-2 are close)
    logits = np.array([5.0, 4.8, -10.0, -11.0, -12.0])
    
    au_func_orig = get_eu(mode="au", k=2)
    
    print(f"\nOriginal logits: {logits}")
    print(f"Top-2 logits: {np.partition(logits, -2)[-2:]}")
    
    print("\nApproach 1: Direct (original - may fail)")
    try:
        au1 = au_func_orig(logits)
        print(f"  AU: {au1:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\nApproach 2: Shift to positive")
    logits_shifted = logits - np.min(logits) + 1.0
    print(f"  Shifted logits: {logits_shifted}")
    print(f"  Top-2 shifted: {np.partition(logits_shifted, -2)[-2:]}")
    try:
        au2 = au_func_orig(logits_shifted)
        print(f"  AU: {au2:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\nApproach 3: Use exp to normalize")
    logits_exp = np.exp(logits - np.max(logits))  # Numerical stability
    logits_exp = logits_exp * 100  # Scale up
    print(f"  Exp-normalized (top-2): {np.partition(logits_exp, -2)[-2:]}")
    try:
        au3 = au_func_orig(logits_exp)
        print(f"  AU: {au3:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\nApproach 4: Only shift top-k values")
    top_k = 2
    top_values = np.partition(logits, -top_k)[-top_k:]
    top_values_shifted = top_values - np.min(top_values) + 1.0
    print(f"  Top-2 original: {top_values}")
    print(f"  Top-2 shifted: {top_values_shifted}")
    # Reconstruct logits with shifted top values
    logits_shifted_topk = logits.copy()
    top_indices = np.argpartition(logits, -top_k)[-top_k:]
    logits_shifted_topk[top_indices] = top_values_shifted
    try:
        au4 = au_func_orig(logits_shifted_topk)
        print(f"  AU: {au4:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_au_calculation()
    test_eu_calculation()
    test_with_real_model_logits()
    test_logit_shift_approach()

