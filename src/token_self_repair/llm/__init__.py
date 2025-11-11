"""LLM client abstractions and providers."""

from .base import LLMClient, TokenLogit
from .mocks import DeterministicMockLLM

# Optional imports - only load if dependencies are available
try:
    from .llama_provider import LlamaProvider, load_llama
    _HAS_LLAMA = True
except ImportError:
    _HAS_LLAMA = False
    LlamaProvider = None
    load_llama = None

try:
    from .openai_provider import OpenAIProvider
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    OpenAIProvider = None

__all__ = [
    "LLMClient",
    "TokenLogit",
    "DeterministicMockLLM",
]

if _HAS_LLAMA:
    __all__.extend(["LlamaProvider", "load_llama"])
if _HAS_OPENAI:
    __all__.append("OpenAIProvider")
