#!/usr/bin/env python3
"""
Deep Neural Network baseline for UCI Superconductor Database.

Provides a fair "deep learning" comparison to Random Forest when crystal
structures are not available (GNNs not applicable).

Architecture: 4-layer feedforward network with dropout regularization.
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Fixed seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class SuperconductorDNN(nn.Module):
    """
    Deep Neural Network for superconductor Tc prediction.
    
    Architecture:
        Input (81 features)
        → Dense(256) + ReLU + Dropout(0.2)
        → Dense(128) + ReLU + Dropout(0.2)
        → Dense(64) + ReLU
        → Dense(1) [Tc prediction]
    
    Design choices:
        - 4 layers: Balance between capacity and overfitting
        - Decreasing width: Funnel information extraction
        - Dropout: Regularization (p=0.2 moderate)
        - ReLU: Standard activation for regression
        - No batch norm: Dataset is pre-scaled (StandardScaler)
    """
    
    def __init__(self, input_dim: int = 81, hidden_dims: Tuple[int, ...] = (256, 128, 64), dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with dropout
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Last hidden layer (no dropout before output)
        layers.extend([
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU()
        ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Xavier/Glorot)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x).squeeze(-1)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_data_splits(data_path: str = "data/raw/train.csv") -> Dict:
    """
    Load and split UCI dataset using the data_loader module.
    
    Returns splits dictionary with tensors.
    """
    import sys
    from pathlib import Path
    
    # Add repo root to path
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root))
    
    from validation.baselines.data_loader import load_uci_superconductor_data, split_data
    
    logger.info("Loading UCI Superconductor Database...")
    X, y, _ = load_uci_superconductor_data(data_path, compute_checksum=True)
    
    logger.info("Splitting data (70/10/20, seed=42)...")
    splits = split_data(X, y, test_size=0.20, val_size=0.10, seed=SEED, scale_features=True)
    
    # Convert to PyTorch tensors
    splits_tensor = {
        "X_train": torch.FloatTensor(splits["X_train"]),
        "y_train": torch.FloatTensor(splits["y_train"]),
        "X_val": torch.FloatTensor(splits["X_val"]),
        "y_val": torch.FloatTensor(splits["y_val"]),
        "X_test": torch.FloatTensor(splits["X_test"]),
        "y_test": torch.FloatTensor(splits["y_test"]),
        "scaler": splits["scaler"],
    }
    
    return splits_tensor


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item() * len(X_batch)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    return avg_loss, preds, targets


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_model_checksum(model: nn.Module) -> str:
    """Compute SHA-256 checksum of model weights."""
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()


def train_dnn(
    splits: Dict,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 15,
    device: str = "cpu"
) -> Dict:
    """
    Train Deep Neural Network on UCI superconductor data.
    
    Args:
        splits: Data splits (X_train, y_train, etc.)
        epochs: Maximum training epochs
        batch_size: Batch size for training
        lr: Learning rate
        patience: Early stopping patience
        device: "cpu" or "cuda"
    
    Returns:
        Dictionary with model, metrics, and training history
    """
    device = torch.device(device)
    
    logger.info("\n" + "="*70)
    logger.info("Training Deep Neural Network Baseline")
    logger.info("="*70)
    
    # Create dataloaders
    train_dataset = TensorDataset(splits["X_train"], splits["y_train"])
    val_dataset = TensorDataset(splits["X_val"], splits["y_val"])
    test_dataset = TensorDataset(splits["X_test"], splits["y_test"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = splits["X_train"].shape[1]
    model = SuperconductorDNN(input_dim=input_dim).to(device)
    
    logger.info(f"Model architecture: {model}")
    logger.info(f"Trainable parameters: {model.count_parameters():,}")
    logger.info(f"Device: {device}")
    
    # Optimizer and loss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    
    start_time = time.time()
    
    logger.info(f"\nTraining for up to {epochs} epochs (early stopping patience={patience})...")
    logger.info(f"Batch size: {batch_size}, Learning rate: {lr}\n")
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
        
        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{epochs}: "
                       f"Train Loss={train_loss:.4f}, "
                       f"Val Loss={val_loss:.4f}, "
                       f"Best Val={best_val_loss:.4f} @ epoch {best_epoch+1}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on all splits
    logger.info(f"\nEvaluating best model (epoch {best_epoch+1})...")
    
    _, train_preds, train_targets = evaluate(model, train_loader, criterion, device)
    _, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
    _, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    
    train_metrics = compute_metrics(train_targets, train_preds)
    val_metrics = compute_metrics(val_targets, val_preds)
    test_metrics = compute_metrics(test_targets, test_preds)
    
    logger.info(f"\n✅ Training complete ({training_time:.2f}s)")
    logger.info(f"   Train: RMSE={train_metrics['rmse']:.2f}K, MAE={train_metrics['mae']:.2f}K, R²={train_metrics['r2']:.3f}")
    logger.info(f"   Val:   RMSE={val_metrics['rmse']:.2f}K, MAE={val_metrics['mae']:.2f}K, R²={val_metrics['r2']:.3f}")
    logger.info(f"   Test:  RMSE={test_metrics['rmse']:.2f}K, MAE={test_metrics['mae']:.2f}K, R²={test_metrics['r2']:.3f}")
    
    # Compute model checksum
    model_checksum = compute_model_checksum(model)
    logger.info(f"\n   Model checksum (SHA-256): {model_checksum[:16]}...")
    
    return {
        "model": model,
        "model_name": "Deep Neural Network",
        "architecture": f"DNN-{input_dim}-256-128-64-1",
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "r2": test_metrics["r2"],
        "training_time_s": training_time,
        "best_epoch": best_epoch + 1,
        "total_epochs": epoch + 1,
        "status": "COMPLETE",
        "hyperparameters": {
            "input_dim": input_dim,
            "hidden_dims": [256, 128, 64],
            "dropout": 0.2,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "early_stopping_patience": patience,
            "seed": SEED,
        },
        "metrics_per_split": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "history": history,
        "model_checksum": model_checksum,
    }


def save_model(model: nn.Module, output_path: Path):
    """Save model weights."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"✅ Model saved: {output_path}")


def save_results(results: Dict, output_path: Path):
    """Save training results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove model object (not JSON serializable)
    results_json = {k: v for k, v in results.items() if k != "model"}
    
    # Convert NumPy types to Python types for JSON serialization
    def convert_to_python(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results_json = convert_to_python(results_json)
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"✅ Results saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Deep Neural Network baseline")
    parser.add_argument("--data-path", default="data/raw/train.csv", help="Path to UCI dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--output-dir", default="validation/artifacts/baselines", help="Output directory")
    args = parser.parse_args()
    
    # Load data
    splits = load_data_splits(args.data_path)
    
    # Train model
    results = train_dnn(
        splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device
    )
    
    # Save model and results
    output_dir = Path(args.output_dir)
    save_model(results["model"], output_dir / "dnn_model.pt")
    save_results(results, output_dir / "dnn_results.json")
    
    logger.info("\n" + "="*70)
    logger.info("✅ DNN baseline training complete!")
    logger.info("="*70)
    logger.info(f"\nFiles saved:")
    logger.info(f"  - {output_dir / 'dnn_model.pt'}")
    logger.info(f"  - {output_dir / 'dnn_results.json'}")


if __name__ == "__main__":
    main()

