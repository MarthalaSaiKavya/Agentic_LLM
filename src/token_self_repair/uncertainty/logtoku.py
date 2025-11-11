"""
LogTokU (Logits-induced Token Uncertainty) implementation.

This module implements the uncertainty decomposition from:
Ma et al. 2025 - "Estimating LLM Uncertainty with Evidence"

Reference implementation from: logtoku/SenU/metrics.py
"""

import numpy as np
from typing import Optional, Dict, List, Iterable, Sequence, TYPE_CHECKING
from scipy.special import softmax, digamma
from dataclasses import dataclass

# Optional torch import - only needed for analyze() method
if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
        _HAS_TORCH = True
    except ImportError:
        _HAS_TORCH = False
        torch = None

from .base import UncertaintyEstimator
from ..types import TokenScore, UncertaintyLevel
from ..utils.token_ops import to_probabilities


@dataclass
class UncertaintyScores:
    """
    Container for uncertainty scores.
    
    Attributes:
        eu: Epistemic Uncertainty (knowledge gap)
        au: Aleatoric Uncertainty (data ambiguity)
        total: Combined uncertainty (EU × AU)
        entropy: Shannon entropy
        token_texts: Optional token strings
    """
    eu: np.ndarray  # Shape: (num_tokens,)
    au: np.ndarray  # Shape: (num_tokens,)
    total: np.ndarray  # Shape: (num_tokens,)
    entropy: np.ndarray  # Shape: (num_tokens,)
    token_texts: Optional[List[str]] = None
    
    @property
    def avg_eu(self) -> float:
        """Average epistemic uncertainty."""
        return float(np.mean(self.eu))
    
    @property
    def avg_au(self) -> float:
        """Average aleatoric uncertainty."""
        return float(np.mean(self.au))
    
    @property
    def avg_total(self) -> float:
        """Average total uncertainty."""
        return float(np.mean(self.total))
    
    @property
    def avg_entropy(self) -> float:
        """Average entropy."""
        return float(np.mean(self.entropy))
    
    @property
    def max_total(self) -> float:
        """Maximum total uncertainty."""
        return float(np.max(self.total))
    
    def get_top_uncertain_indices(self, k: int = 5) -> np.ndarray:
        """
        Get indices of top-k most uncertain tokens.
        
        Args:
            k: Number of top uncertain tokens
        
        Returns:
            Indices sorted by total uncertainty (descending)
        """
        return np.argsort(self.total)[-k:][::-1]
    
    def get_uncertain_tokens(self, threshold: float = 0.5) -> List[int]:
        """
        Get indices of tokens exceeding uncertainty threshold.
        
        Args:
            threshold: Uncertainty threshold (0-1 range for normalized)
        
        Returns:
            List of token indices
        """
        return np.where(self.total > threshold)[0].tolist()


class LogTokUEstimator(UncertaintyEstimator):
    """
    LogTokU uncertainty estimator.
    
    Decomposes uncertainty into:
    - EU (Epistemic): Measures knowledge gaps (lacks training examples)
    - AU (Aleatoric): Measures data ambiguity (multiple valid interpretations)
    
    The total LogTokU score is: LogTokU = EU × AU
    
    High EU → Model hasn't seen similar patterns (exploration needed)
    High AU → Multiple valid interpretations exist (refinement needed)
    
    Example:
        >>> estimator = LogTokUEstimator(k=2)
        >>> scores = estimator.analyze(logits_tensor)
        >>> print(f"Avg EU: {scores.avg_eu:.4f}")
        >>> print(f"Avg AU: {scores.avg_au:.4f}")
    """
    
    def __init__(self, k: int = 2):
        """
        Initialize LogTokU estimator.
        
        Args:
            k: Top-k parameter for uncertainty calculation (default: 2)
               Higher k considers more top tokens in calculation
        """
        self.k = k
    
    def calculate_entropy(self, logits: np.ndarray) -> float:
        """
        Calculate Shannon entropy from logits.
        
        Args:
            logits: Raw logits for single token (vocab_size,)
        
        Returns:
            Entropy value
        """
        probs = softmax(logits)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def calculate_eu(self, logits: np.ndarray) -> float:
        """
        Calculate Epistemic Uncertainty (EU).
        
        EU measures knowledge gaps. Higher values indicate the model
        lacks similar training examples.
        
        Formula: k / (sum(max(0, top_k_logits)) + k)
        
        Args:
            logits: Raw logits for single token (vocab_size,)
        
        Returns:
            EU score (higher = more epistemic uncertainty)
        """
        if len(logits) < self.k:
            raise ValueError(f"Logits length {len(logits)} < k={self.k}")
        
        # Get top-k logit values
        top_k_indices = np.argpartition(logits, -self.k)[-self.k:]
        top_k_values = logits[top_k_indices]
        
        # Calculate EU
        eu = self.k / (np.sum(np.maximum(0, top_k_values)) + self.k)
        
        return eu
    
    def calculate_au(self, logits: np.ndarray) -> float:
        """
        Calculate Aleatoric Uncertainty (AU).
        
        AU measures data ambiguity. Higher values indicate multiple
        valid interpretations exist in the training data.
        
        Uses Dirichlet distribution with digamma function:
        AU = -sum((α_i / α_0) * (ψ(α_i + 1) - ψ(α_0 + 1)))
        
        Where α are the top-k logits.
        
        Args:
            logits: Raw logits for single token (vocab_size,)
        
        Returns:
            AU score (higher = more aleatoric uncertainty)
        """
        if len(logits) < self.k:
            raise ValueError(f"Logits length {len(logits)} < k={self.k}")
        
        # Get top-k logit values and convert to positive evidence
        top_k_values = np.partition(logits, -self.k)[-self.k:]
        evidence = np.maximum(top_k_values, 0) + 1e-3
        
        # Dirichlet parameters (ensure strictly positive)
        alpha = np.array([evidence])
        alpha_0 = alpha.sum(axis=1, keepdims=True)

        # Digamma calculations
        psi_alpha_k_plus_1 = digamma(alpha + 1)
        psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
        
        # AU formula
        result = -(alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
        au = result.sum(axis=1)[0]
        
        return au
    
    def analyze_single_token(self, logits: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Analyze uncertainty for a single token.
        
        Args:
            logits: Logits for single token (vocab_size,)
        
        Returns:
            Dictionary with EU, AU, LogTokU, Entropy or None if invalid
        """
        try:
            eu = self.calculate_eu(logits)
            au = self.calculate_au(logits)
            
            # Check for invalid values
            if np.isnan(au) or au <= 0:
                return None
            
            logtoku = eu * au
            entropy = self.calculate_entropy(logits)
            
            return {
                'EU': eu,
                'AU': au,
                'LogTokU': logtoku,
                'Entropy': entropy
            }
        except (ValueError, RuntimeError) as e:
            return None
    
    def analyze(
        self,
        logits,
        token_texts: Optional[List[str]] = None
    ) -> Optional[UncertaintyScores]:
        """
        Analyze uncertainty for multiple tokens.
        
        This is the main entry point for uncertainty estimation.
        
        Args:
            logits: Logits tensor of shape (num_tokens, vocab_size) (torch.Tensor or numpy array)
            token_texts: Optional list of token strings for reference
        
        Returns:
            UncertaintyScores object or None if analysis fails
        
        Example:
            >>> from token_self_repair.llm import LlamaProvider
            >>> provider = LlamaProvider("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            >>> tokens, logits = provider.generate_with_logits("Hello world")
            >>> 
            >>> estimator = LogTokUEstimator(k=2)
            >>> scores = estimator.analyze(logits)
            >>> 
            >>> print(f"Average EU: {scores.avg_eu:.4f}")
            >>> print(f"Average AU: {scores.avg_au:.4f}")
            >>> print(f"Average LogTokU: {scores.avg_total:.4f}")
            >>> 
            >>> # Get most uncertain tokens
            >>> top_5 = scores.get_top_uncertain_indices(k=5)
            >>> print(f"Most uncertain token indices: {top_5}")
        """
        # Convert to numpy if needed
        if _HAS_TORCH and torch is not None and isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        
        # Ensure 2D
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        
        # Analyze each token
        token_metrics = []
        for i in range(logits.shape[0]):
            metrics = self.analyze_single_token(logits[i])
            if metrics:
                token_metrics.append(metrics)
        
        if not token_metrics:
            return None
        
        # Aggregate results
        eus = np.array([m['EU'] for m in token_metrics])
        aus = np.array([m['AU'] for m in token_metrics])
        logtokus = np.array([m['LogTokU'] for m in token_metrics])
        entropies = np.array([m['Entropy'] for m in token_metrics])
        
        return UncertaintyScores(
            eu=eus,
            au=aus,
            total=logtokus,
            entropy=entropies,
            token_texts=token_texts
        )
    
    # ------------------------------------------------------------------
    # Integration with UncertaintyEstimator interface
    # ------------------------------------------------------------------
    def score(
        self,
        tokens: Iterable[str],
        logits: Iterable[Sequence[float]],
    ) -> Iterable[TokenScore]:
        """
        Compute TokenScore objects for streaming pipelines.

        Args:
            tokens: Generated token strings
            logits: Sequence of logit vectors (aligned with tokens)

        Yields:
            TokenScore for each token
        """

        token_list = list(tokens)
        logit_list = [np.asarray(l, dtype=np.float32) for l in logits]

        if not token_list or not logit_list:
            return []

        # Analyze uncertainty using existing pipeline
        scores = self.analyze(np.stack(logit_list, axis=0), token_texts=token_list)
        if scores is None:
            return []

        def classify(total: float) -> UncertaintyLevel:
            if total < 0.2:
                return UncertaintyLevel.HIGH_CONFIDENCE
            if total < 0.5:
                return UncertaintyLevel.MODERATE
            return UncertaintyLevel.LOW

        token_scores: List[TokenScore] = []
        for idx, token in enumerate(token_list):
            logits_vec = logit_list[idx]
            probs = to_probabilities(logits_vec)
            probability = max(probs) if probs else 0.0
            max_logit = float(np.max(logits_vec)) if logits_vec.size > 0 else 0.0
            total_uncertainty = float(scores.total[idx])

            token_scores.append(
                TokenScore(
                    token=token,
                    logit=max_logit,
                    probability=float(probability),
                    entropy=float(scores.entropy[idx]),
                    aleatoric=float(scores.au[idx]),
                    epistemic=float(scores.eu[idx]),
                    total_uncertainty=total_uncertainty,
                    level=classify(total_uncertainty),
                )
            )

        return token_scores

    def get_uncertainty_type(self, eu: float, au: float) -> str:
        """
        Classify uncertainty type based on EU and AU values.
        
        Args:
            eu: Epistemic uncertainty
            au: Aleatoric uncertainty
        
        Returns:
            Uncertainty type: 'epistemic', 'aleatoric', 'both', or 'low'
        """
        eu_threshold = 0.5
        au_threshold = 0.5
        
        high_eu = eu > eu_threshold
        high_au = au > au_threshold
        
        if high_eu and high_au:
            return 'both'
        elif high_eu:
            return 'epistemic'
        elif high_au:
            return 'aleatoric'
        else:
            return 'low'


# Convenience functions for quick testing

def quick_analyze(logits, k: int = 2) -> UncertaintyScores:
    """
    Quick analysis with default settings.
    
    Args:
        logits: Logits tensor (num_tokens, vocab_size)
        k: Top-k parameter
    
    Returns:
        UncertaintyScores object
    """
    estimator = LogTokUEstimator(k=k)
    return estimator.analyze(logits)


def analyze_generation(
    provider,
    prompt: str,
    max_tokens: int = 50
) -> Dict[str, any]:
    """
    End-to-end analysis: generate and analyze uncertainty.
    
    Args:
        provider: LlamaProvider instance
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
    
    Returns:
        Dictionary with generation and uncertainty info
    
    Example:
        >>> from token_self_repair.llm import load_llama
        >>> provider = load_llama()
        >>> result = analyze_generation(provider, "What is 2+2?")
        >>> print(result['avg_eu'], result['avg_au'])
    """
    # Generate with logits
    tokens, logits = provider.generate_with_logits(prompt, max_tokens)
    
    # Decode tokens
    response = provider.tokenizer.decode(tokens, skip_special_tokens=True)
    token_texts = [provider.tokenizer.decode([t]) for t in tokens]
    
    # Analyze uncertainty
    estimator = LogTokUEstimator(k=2)
    scores = estimator.analyze(logits, token_texts=token_texts)
    
    if scores is None:
        return None
    
    return {
        'prompt': prompt,
        'response': response,
        'tokens': token_texts,
        'avg_eu': scores.avg_eu,
        'avg_au': scores.avg_au,
        'avg_logtoku': scores.avg_total,
        'avg_entropy': scores.avg_entropy,
        'max_logtoku': scores.max_total,
        'num_tokens': len(scores.eu),
        'scores': scores
    }
