"""Central configuration for the token-level self-repair framework."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class Thresholds:
    """Confidence thresholds that control self-repair activation."""

    high_confidence: float = 0.8
    moderate_confidence: float = 0.6
    low_confidence: float = 0.4
    # For smaller models (e.g., 8B) we want the repair loop to be a bit more
    # sensitive, so trigger repairs once total uncertainty exceeds ~0.25.
    repair_activation_uncertainty: float = 0.25


# Constants for LangGraph agentic workflow
UNCERTAINTY_THRESHOLD: float = 0.5
MAX_REPAIR_ATTEMPTS: int = 3


@dataclass(slots=True)
class ProjectConfig:
    """Container for project-wide configuration values."""

    vocab_size: int = 32000
    max_self_repairs: int = 2
    thresholds: Thresholds = field(default_factory=Thresholds)
    messaging_channels: Dict[str, bool] = field(
        default_factory=lambda: {"console": True, "event_bus": False}
    )
