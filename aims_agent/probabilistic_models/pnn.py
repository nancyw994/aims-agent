"""
Probabilistic Neural Network (PNN) for uncertainty quantification.

The PNN predicts both mean (mu) and variance (sigma^2) for each input,
representing the output as a Gaussian distribution: y | x ~ N(mu(x), sigma^2(x))

This enables native uncertainty quantification without ensembles.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


class ProbabilisticNN(nn.Module):
    """
    Probabilistic Neural Network that predicts Gaussian distributions.
    
    Outputs:
        mu: Mean of the predictive distribution
        var: Variance of the predictive distribution (heteroscedastic)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.1
    ):
        """
        Initialize Probabilistic NN.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build backbone
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Two heads: one for mean, one for log variance
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            mu: Predicted mean, shape (batch_size, 1)
            var: Predicted variance, shape (batch_size, 1)
        """
        # Shared backbone
        h = self.backbone(x)
        
        # Mean prediction
        mu = self.mu_head(h)
        
        # Log variance prediction (for numerical stability)
        log_var = self.log_var_head(h)
        
        # Convert to variance with lower bound for stability
        var = torch.exp(log_var) + 1e-6
        
        return mu, var


class ProbabilisticNNWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for ProbabilisticNN.
    
    This allows the model to be used with sklearn's API while providing
    additional methods for distribution prediction.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: str = 'cpu',
        random_state: int = 42,
        verbose: bool = False
    ):
        """
        Initialize PNN wrapper.
        
        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout_rate: Dropout probability
            epochs: Maximum training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for Adam optimizer
            weight_decay: L2 regularization strength
            patience: Early stopping patience
            device: 'cpu' or 'cuda'
            random_state: Random seed
            verbose: Print training progress
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.input_dim_ = None
        self.training_history_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the probabilistic neural network.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
        
        Returns:
            self
        """
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset
        
        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Convert to numpy if needed
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        
        # Store input dimension
        self.input_dim_ = X.shape[1]
        
        # Standardize inputs and outputs
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create data loader
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Initialize model
        self.model = ProbabilisticNN(
            input_dim=self.input_dim_,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate
        )
        
        device = torch.device(self.device)
        self.model = self.model.to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        self.training_history_ = []
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Forward pass
                mu, var = self.model(X_batch)
                
                # Compute NLL loss
                loss = self._gaussian_nll_loss(y_batch, mu, var)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=1.0
                )
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            self.training_history_.append(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict mean values (sklearn-compatible).
        
        Args:
            X: Test features, shape (n_samples, n_features)
        
        Returns:
            Predicted means, shape (n_samples,)
        """
        dist = self.predict_distribution(X)
        return dist['mu']
    
    def predict_distribution(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        Predict full distribution (mean, std, confidence intervals).
        
        Args:
            X: Test features, shape (n_samples, n_features)
        
        Returns:
            Dictionary with:
                - 'mu': predicted means
                - 'std': predicted standard deviations
                - 'var': predicted variances
                - 'lower_95': lower bound of 95% CI
                - 'upper_95': upper bound of 95% CI
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        
        # Convert to numpy if needed
        X = np.asarray(X)
        
        # Standardize input
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            mu_scaled, var_scaled = self.model(X_tensor)
            std_scaled = torch.sqrt(var_scaled)
        
        # Convert to numpy
        mu_scaled = mu_scaled.cpu().numpy().flatten()
        std_scaled = std_scaled.cpu().numpy().flatten()
        
        # Inverse transform mean
        mu = self.scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).flatten()
        
        # Scale std (not shift!)
        std = std_scaled * self.scaler_y.scale_[0]
        var = std ** 2
        
        # 95% confidence intervals
        lower_95 = mu - 1.96 * std
        upper_95 = mu + 1.96 * std
        
        return {
            'mu': mu,
            'std': std,
            'var': var,
            'lower_95': lower_95,
            'upper_95': upper_95
        }
    
    @staticmethod
    def _gaussian_nll_loss(
        y_true: torch.Tensor, 
        mu: torch.Tensor, 
        var: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood loss.
        
        Args:
            y_true: True values
            mu: Predicted means
            var: Predicted variances
            eps: Small constant for numerical stability
        
        Returns:
            NLL loss (scalar)
        """
        # Clamp variance for stability
        var = torch.clamp(var, min=eps, max=1e6)
        
        # NLL = 0.5 * (log(var) + (y - mu)^2 / var)
        loss = 0.5 * torch.log(var) + 0.5 * (y_true - mu) ** 2 / var
        
        return loss.mean()
    
    def get_training_history(self) -> list[float]:
        """Get training loss history."""
        return self.training_history_


__all__ = ["ProbabilisticNN", "ProbabilisticNNWrapper"]
