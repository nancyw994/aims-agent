"""
Loss functions for probabilistic models.
"""

import torch


def gaussian_nll_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.
    
    This is the standard loss for training probabilistic models that
    predict Gaussian distributions.
    
    Args:
        y_true: True values, shape (batch_size, 1)
        mu: Predicted means, shape (batch_size, 1)
        var: Predicted variances, shape (batch_size, 1)
        eps: Small constant for numerical stability
    
    Returns:
        NLL loss (scalar)
    
    Formula:
        NLL = 0.5 * log(2π * var) + 0.5 * (y - mu)^2 / var
            = 0.5 * log(var) + 0.5 * (y - mu)^2 / var + constant
    """
    # Clamp variance to prevent numerical issues
    var = torch.clamp(var, min=eps, max=1e6)
    
    # Compute NLL (ignoring constant term)
    loss = 0.5 * torch.log(var) + 0.5 * (y_true - mu) ** 2 / var
    
    return loss.mean()


def heteroscedastic_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    lambda_reg: float = 0.01,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Heteroscedastic loss with regularization to prevent variance collapse.
    
    This loss adds a regularization term to encourage the model to produce
    reasonable uncertainty estimates.
    
    Args:
        y_true: True values
        mu: Predicted means
        log_var: Predicted log variances
        lambda_reg: Regularization strength
        eps: Small constant
    
    Returns:
        Total loss (NLL + regularization)
    """
    var = torch.exp(log_var) + eps
    
    # NLL term
    nll = 0.5 * log_var + 0.5 * (y_true - mu) ** 2 / var
    
    # Regularization: penalize too small variances
    # This prevents the model from being overconfident
    reg = lambda_reg * torch.mean(torch.exp(-log_var))
    
    return nll.mean() + reg


__all__ = ["gaussian_nll_loss", "heteroscedastic_loss"]
