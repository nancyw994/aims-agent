"""
Trainer for probabilistic models with full training pipeline.

This module provides a high-level training interface for probabilistic neural networks,
including data preparation, training loop, and distribution prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from aims_agent.probabilistic_models.pnn import ProbabilisticNN
from aims_agent.probabilistic_models.losses import gaussian_nll_loss


class ProbabilisticTrainer:
    """
    High-level trainer for probabilistic neural networks.
    
    Handles data preparation, model training, early stopping,
    and distribution prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 200,
        patience: int = 15,
        val_split: float = 0.2,
        device: str | None = None,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize probabilistic trainer.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout_rate: Dropout probability
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
            batch_size: Training batch size
            epochs: Maximum epochs
            patience: Early stopping patience
            val_split: Validation set proportion
            device: 'cpu', 'cuda', or None (auto-detect)
            random_state: Random seed
            verbose: Print training progress
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.verbose = verbose
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Training history
        self.train_losses_ = []
        self.val_losses_ = []
        self.best_epoch_ = 0
        self.is_fitted_ = False
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[DataLoader, DataLoader | None]:
        """
        Prepare and split data for training.
        
        Args:
            X: Features, shape (n_samples, n_features)
            y: Targets, shape (n_samples,)
        
        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader (or None)
        """
        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Split data if validation set requested
        if self.val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.val_split,
                random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Standardize
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(
            y_train.reshape(-1, 1)
        ).flatten()
        
        # Create train loader
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train_scaled).unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Create validation loader if needed
        val_loader = None
        if X_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(
                y_val.reshape(-1, 1)
            ).flatten()
            
            X_val_tensor = torch.FloatTensor(X_val_scaled)
            y_val_tensor = torch.FloatTensor(y_val_scaled).unsqueeze(1)
            
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        return train_loader, val_loader
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict[str, Any]:
        """
        Train the probabilistic model.
        
        Args:
            X: Features, shape (n_samples, n_features)
            y: Targets, shape (n_samples,)
        
        Returns:
            Training info dictionary with losses and best epoch
        """
        # Prepare data
        train_loader, val_loader = self.prepare_data(X, y)
        
        # Initialize model
        self.model = ProbabilisticNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        if self.verbose:
            print(f"\nTraining Probabilistic NN on {self.device}")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.epochs):
            # Train phase
            train_loss = self._train_epoch(train_loader)
            self.train_losses_.append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.val_losses_.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_epoch_ = epoch
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state_ = {
                        k: v.cpu().clone() 
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Best: {best_val_loss:.4f}")
                
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        print(f"Best epoch: {self.best_epoch_+1} with val loss: {best_val_loss:.4f}")
                    break
            else:
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{self.epochs} | Train Loss: {train_loss:.4f}")
        
        # Load best model if validation was used
        if val_loader is not None and hasattr(self, 'best_model_state_'):
            self.model.load_state_dict(self.best_model_state_)
        
        self.is_fitted_ = True
        
        return {
            'train_losses': self.train_losses_,
            'val_losses': self.val_losses_,
            'best_epoch': self.best_epoch_,
            'final_train_loss': self.train_losses_[-1],
            'final_val_loss': self.val_losses_[-1] if self.val_losses_ else None,
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            mu, var = self.model(X_batch)
            loss = gaussian_nll_loss(y_batch, mu, var)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                mu, var = self.model(X_batch)
                loss = gaussian_nll_loss(y_batch, mu, var)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict_distribution(
        self,
        X: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Predict distribution for test data.
        
        Args:
            X: Test features, shape (n_samples, n_features)
        
        Returns:
            Dictionary with:
                - 'mu': predicted means
                - 'std': predicted standard deviations
                - 'var': predicted variances
                - 'lower_95': 95% CI lower bound
                - 'upper_95': 95% CI upper bound
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        # Standardize input
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            mu_scaled, var_scaled = self.model(X_tensor)
            std_scaled = torch.sqrt(var_scaled)
        
        # Convert to numpy
        mu_scaled = mu_scaled.cpu().numpy().flatten()
        std_scaled = std_scaled.cpu().numpy().flatten()
        
        # Inverse transform
        mu = self.scaler_y.inverse_transform(
            mu_scaled.reshape(-1, 1)
        ).flatten()
        
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
    
    def save(self, path: str | Path):
        """Save model and scalers."""
        import pickle
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate,
            },
            'training_history': {
                'train_losses': self.train_losses_,
                'val_losses': self.val_losses_,
                'best_epoch': self.best_epoch_,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str | Path, device: str | None = None):
        """Load model and scalers."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Create trainer
        config = state['config']
        trainer = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout_rate=config['dropout_rate'],
            device=device
        )
        
        # Load model
        trainer.model = ProbabilisticNN(**config).to(trainer.device)
        trainer.model.load_state_dict(state['model_state'])
        
        # Load scalers
        trainer.scaler_X = state['scaler_X']
        trainer.scaler_y = state['scaler_y']
        
        # Load history
        trainer.train_losses_ = state['training_history']['train_losses']
        trainer.val_losses_ = state['training_history']['val_losses']
        trainer.best_epoch_ = state['training_history']['best_epoch']
        
        trainer.is_fitted_ = True
        
        return trainer


__all__ = ["ProbabilisticTrainer"]
