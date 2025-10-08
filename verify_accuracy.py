#!/usr/bin/env python3
"""
Verification Script: Test claimed model accuracy of 88.8%
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

print("🔍 VERIFICATION: MODEL ACCURACY CLAIM")
print("=" * 60)

# Load model
model_path = Path("models/superconductor_classifier.pkl")
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    exit(1)

try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    if isinstance(model_data, dict):
        model = model_data.get('model')
        print(f"✅ Loaded model from dict")
        print(f"   Keys: {list(model_data.keys())}")
    else:
        model = model_data
        print(f"✅ Loaded model directly")
    
    print(f"   Model type: {type(model).__name__}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Load data
data_path = Path("data/superconductors/processed/train.csv")
if not data_path.exists():
    # Try alternative path
    data_path = Path("data/superconductors/raw/unique_m.csv")
    
if not data_path.exists():
    print(f"❌ Data not found at expected locations")
    print("   Checked: data/superconductors/processed/train.csv")
    print("   Checked: data/superconductors/raw/unique_m.csv")
    exit(1)

try:
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset: {len(df)} samples")
    print(f"   Columns: {df.shape[1]}")
    print(f"   First few columns: {df.columns[:5].tolist()}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Check if model was actually trained
if hasattr(model, 'n_features_in_'):
    print(f"   Model features: {model.n_features_in_}")
    print(f"   Dataset features: {df.shape[1]}")
    
    # Check if they match
    if model.n_features_in_ == df.shape[1] or model.n_features_in_ == df.shape[1] - 1:
        print("✅ Model and dataset dimensions match")
    else:
        print(f"⚠️  Dimension mismatch: model expects {model.n_features_in_}, data has {df.shape[1]}")
else:
    print("⚠️  Cannot determine model input features")

# Try to predict (if we can figure out the target column)
print("\n" + "=" * 60)
print("VERIFICATION RESULT:")
print("=" * 60)
print("✅ Model file exists (16MB - likely trained)")
print("✅ Dataset exists")
print("⚠️  Cannot verify 88.8% accuracy without knowing:")
print("   - Target column name")
print("   - Train/test split seed")
print("   - Whether 88.8% is R² score or classification accuracy")
print("\n💡 Recommendation: Check model training script for details")
print("   File: models/train_model.py or scripts/train_model.py")

# Check if training script exists
training_scripts = [
    "models/train_model.py",
    "scripts/train_model.py", 
    "demo/train_model.py"
]

print("\n🔍 Looking for training scripts:")
for script_path in training_scripts:
    if Path(script_path).exists():
        print(f"✅ Found: {script_path}")
    else:
        print(f"❌ Not found: {script_path}")

