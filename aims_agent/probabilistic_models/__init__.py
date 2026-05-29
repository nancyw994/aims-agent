"""
Probabilistic models for uncertainty quantification.

This module provides neural network models that output predictive distributions
rather than point predictions, enabling native uncertainty quantification.
"""

from aims_agent.probabilistic_models.pnn import (
    ProbabilisticNN,
    ProbabilisticNNWrapper,
)
from aims_agent.probabilistic_models.losses import (
    gaussian_nll_loss,
    heteroscedastic_loss,
)

__all__ = [
    "ProbabilisticNN",
    "ProbabilisticNNWrapper",
    "gaussian_nll_loss",
    "heteroscedastic_loss",
]
