"""Quick diagnostic for DKL training issue."""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase10_gp_active_learning.models.dkl_model import create_dkl_model
from phase10_gp_active_learning.models.botorch_dkl import BoTorchDKL

print("=" * 70)
print("DKL TRAINING DIAGNOSTIC")
print("=" * 70)

# Generate synthetic data
np.random.seed(42)
torch.manual_seed(42)

n_train = 100
n_test = 50
n_features = 81

X_train = np.random.randn(n_train, n_features)
y_train = X_train[:, 0] * 10 + X_train[:, 1] * 5 + np.random.randn(n_train) * 0.5
X_test = np.random.randn(n_test, n_features)
y_test = X_test[:, 0] * 10 + X_test[:, 1] * 5

print(f"\n📊 Data:")
print(f"   Train: {X_train.shape}, y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"   Test: {X_test.shape}, y range: [{y_test.min():.2f}, {y_test.max():.2f}]")

# Train DKL
print(f"\n🔧 Training DKL...")
dkl = create_dkl_model(
    X_train, y_train,
    input_dim=n_features,
    n_epochs=50,
    lr=0.001,
    verbose=True  # Enable verbose to see training
)

# Check feature extractor
print(f"\n🔍 Feature Extractor:")
print(f"   Training mode: {dkl.feature_extractor.training}")
X_test_t = torch.tensor(X_test, dtype=torch.float64)
z = dkl.feature_extractor(X_test_t)
print(f"   Feature range: [{z.min():.4f}, {z.max():.4f}]")
print(f"   Feature std: {z.std():.4f}")

# Check GP predictions on features
print(f"\n🔍 Latent MVN:")
mvn = dkl.latent_mvn(X_test_t, observation_noise=False)
print(f"   MVN mean range: [{mvn.mean.min():.4f}, {mvn.mean.max():.4f}]")
print(f"   MVN variance range: [{mvn.variance.min():.4f}, {mvn.variance.max():.4f}]")

# Check DKL predictions directly
print(f"\n🔍 DKL Predictions:")
y_pred, y_std = dkl.predict(X_test)
print(f"   Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
print(f"   Std range: [{y_std.min():.4f}, {y_std.max():.4f}]")

# Check BoTorch wrapper
print(f"\n🔍 BoTorch Wrapper:")
model = BoTorchDKL(dkl)
posterior = model.posterior(X_test_t)
print(f"   Posterior mean range: [{posterior.mean.min():.4f}, {posterior.mean.max():.4f}]")
print(f"   Posterior variance range: [{posterior.variance.min():.4f}, {posterior.variance.max():.4f}]")

# Compute RMSE
from sklearn.metrics import mean_squared_error
rmse_direct = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_botorch = np.sqrt(mean_squared_error(y_test, posterior.mean.squeeze().detach().cpu().numpy()))

print(f"\n📈 RMSE:")
print(f"   Direct prediction: {rmse_direct:.4f}")
print(f"   BoTorch prediction: {rmse_botorch:.4f}")
print(f"   Baseline (mean): {np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))):.4f}")

# Diagnosis
print(f"\n🔬 DIAGNOSIS:")
if z.std() < 0.01:
    print("   ❌ Feature extractor outputs nearly constant!")
    print("   → NN parameters may not be updating during training")
elif mvn.mean.std() < 0.01:
    print("   ❌ GP predictions nearly constant!")
    print("   → GP not learning from features")
elif y_pred.std() < 1.0:
    print("   ⚠️  Predictions have low variance")
    print("   → Model may be underfitting")
else:
    print("   ✅ Model appears to be working (has variance)")
    print("   → Issue may be in benchmark loop, not model itself")

print("\n" + "=" * 70)

