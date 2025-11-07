"""
Simple test script to verify LogTokU setup and functionality.
This script tests the core LogTokU uncertainty calculation without requiring
a full model setup.
"""

import numpy as np
from scipy.special import softmax, digamma
import sys
import os

# Add SenU to path to import metrics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SenU'))
from metrics import get_eu

def topk(arr, k):
    """Get top-k values from array"""
    indices = np.argpartition(arr, -k)[-k:]
    values = arr[indices]
    return values, indices

def test_logtoku_calculation():
    """Test LogTokU uncertainty calculation with sample logits"""
    print("Testing LogTokU uncertainty calculation...")
    print("-" * 50)
    
    # Create sample logits (simulating model output)
    # High uncertainty case: similar logits for multiple tokens
    high_uncertainty_logits = np.array([2.1, 2.0, 1.9, 1.8, 1.7, -5.0, -4.0])
    
    # Low uncertainty case: one token has much higher logit
    low_uncertainty_logits = np.array([10.0, 2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    
    print("\n1. Testing EU (Epistemic Uncertainty) calculation:")
    eu_func = get_eu(mode="eu", k=2)
    
    eu_high = eu_func(high_uncertainty_logits)
    eu_low = eu_func(low_uncertainty_logits)
    
    print(f"   High uncertainty logits - EU: {eu_high:.4f}")
    print(f"   Low uncertainty logits - EU: {eu_low:.4f}")
    print(f"   (Higher EU indicates higher uncertainty)")
    
    print("\n2. Testing AU (Aleatoric Uncertainty) calculation:")
    au_func = get_eu(mode="au", k=2)
    
    au_high = au_func(high_uncertainty_logits)
    au_low = au_func(low_uncertainty_logits)
    
    print(f"   High uncertainty logits - AU: {au_high:.4f}")
    print(f"   Low uncertainty logits - AU: {au_low:.4f}")
    print(f"   (Higher AU indicates higher uncertainty)")
    
    print("\n3. Testing LogTokU (EU * AU) combination:")
    logtoku_high = eu_high * au_high
    logtoku_low = eu_low * au_low
    
    print(f"   High uncertainty logits - LogTokU: {logtoku_high:.4f}")
    print(f"   Low uncertainty logits - LogTokU: {logtoku_low:.4f}")
    print(f"   (Higher LogTokU indicates higher uncertainty)")
    
    print("\n4. Testing entropy for comparison:")
    def calculate_entropy(logits):
        probs = softmax(logits)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
    
    entropy_high = calculate_entropy(high_uncertainty_logits)
    entropy_low = calculate_entropy(low_uncertainty_logits)
    
    print(f"   High uncertainty logits - Entropy: {entropy_high:.4f}")
    print(f"   Low uncertainty logits - Entropy: {entropy_low:.4f}")
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"High uncertainty case has LogTokU: {logtoku_high:.4f}")
    print(f"Low uncertainty case has LogTokU: {logtoku_low:.4f}")
    
    if logtoku_high > logtoku_low:
        print("\n✓ Test PASSED: LogTokU correctly identifies high uncertainty")
    else:
        print("\n✗ Test FAILED: LogTokU did not correctly identify uncertainty")
    
    return logtoku_high > logtoku_low

def test_imports():
    """Test that all required packages are installed"""
    print("\nTesting package imports...")
    print("-" * 50)
    
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ torch: Not installed - {e}")
        return False
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers: Not installed - {e}")
        return False
    
    try:
        import datasets
        print(f"✓ datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"✗ datasets: Not installed - {e}")
        return False
    
    try:
        import numpy
        print(f"✓ numpy: {numpy.__version__}")
    except ImportError as e:
        print(f"✗ numpy: Not installed - {e}")
        return False
    
    try:
        import scipy
        print(f"✓ scipy: {scipy.__version__}")
    except ImportError as e:
        print(f"✗ scipy: Not installed - {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("LogTokU Setup Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test LogTokU calculation
        calculation_ok = test_logtoku_calculation()
        
        if calculation_ok:
            print("\n" + "=" * 50)
            print("✓ All tests passed! LogTokU is set up correctly.")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("✗ Calculation test failed.")
            print("=" * 50)
            sys.exit(1)
    else:
        print("\n" + "=" * 50)
        print("✗ Some required packages are missing.")
        print("Please install dependencies: pip install -r requirements.txt")
        print("=" * 50)
        sys.exit(1)

