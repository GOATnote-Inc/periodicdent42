"""
Test Active Learning Under Different Conditions

Find WHERE active learning works by testing multiple scenarios.
Volume negates luck - run many experiments to find success conditions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
from pathlib import Path

def test_condition(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    model_type: str = "rf",
    iterations: int = 20
) -> dict:
    """Test one condition"""
    
    # Subsample features
    if n_features < X.shape[1]:
        feature_idx = np.random.RandomState(42).choice(X.shape[1], n_features, replace=False)
        X = X[:, feature_idx]
    
    # Split
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), train_size=0.8, random_state=42
    )
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Further split train into initial + pool
    initial_idx, pool_idx = train_test_split(
        np.arange(len(X_train)), train_size=100, random_state=42
    )
    
    current_train = set(initial_idx)
    current_pool = set(pool_idx)
    
    results = {"random": [], "uncertainty": []}
    
    for iteration in range(iterations):
        # Train model
        if model_type == "rf":
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        else:
            model = Ridge(alpha=1.0)
        
        train_indices = list(current_train)
        model.fit(X_train[train_indices], y_train[train_indices])
        
        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Random selection
        if len(current_pool) > 0:
            pool_list = list(current_pool)
            batch_size = min(10, len(pool_list))
            selected_random = set(np.random.RandomState(42 + iteration).choice(pool_list, batch_size, replace=False))
            
            # Uncertainty selection
            if hasattr(model, 'estimators_'):
                tree_preds = np.array([tree.predict(X_train[pool_list]) for tree in model.estimators_])
                uncertainties = np.std(tree_preds, axis=0)
                selected_uncertainty = set([pool_list[i] for i in np.argsort(uncertainties)[-batch_size:]])
            else:
                selected_uncertainty = selected_random
            
            # Update for next iteration (use uncertainty for both)
            current_train.update(selected_uncertainty)
            current_pool.difference_update(selected_uncertainty)
        
        results["uncertainty"].append(float(rmse))
        results["random"].append(float(rmse))
    
    # Calculate improvement
    initial_rmse = results["uncertainty"][0]
    final_rmse = results["uncertainty"][-1]
    improvement = (initial_rmse - final_rmse) / initial_rmse * 100
    
    return {
        "name": name,
        "n_features": n_features,
        "model_type": model_type,
        "initial_rmse": initial_rmse,
        "final_rmse": final_rmse,
        "improvement_pct": improvement,
        "uncertainty_results": results["uncertainty"]
    }


def main():
    print("="*70)
    print("TESTING ACTIVE LEARNING UNDER DIFFERENT CONDITIONS")
    print("="*70)
    print("\nObjective: Find WHERE active learning works")
    print("Strategy: Test multiple conditions (volume negates luck)\n")
    
    # Load data
    df = pd.read_csv("data/superconductors/raw/train.csv")
    feature_cols = [c for c in df.columns if c != "critical_temp"]
    X = df[feature_cols].values
    y = df["critical_temp"].values
    
    print(f"Dataset: {len(df)} samples, {X.shape[1]} features\n")
    
    # Test conditions
    conditions = [
        ("Baseline (81 features, RF)", 81, "rf"),
        ("Reduced features (20, RF)", 20, "rf"),
        ("Minimal features (10, RF)", 10, "rf"),
        ("Ultra-minimal (5, RF)", 5, "rf"),
        ("Linear model (20 features)", 20, "linear"),
        ("Linear model (10 features)", 10, "linear"),
    ]
    
    results = []
    
    for name, n_features, model_type in conditions:
        print(f"\nTesting: {name}")
        print("-" * 70)
        
        result = test_condition(
            name=name,
            X=X,
            y=y,
            n_features=n_features,
            model_type=model_type,
            iterations=20
        )
        
        results.append(result)
        
        print(f"  Initial RMSE: {result['initial_rmse']:.2f}K")
        print(f"  Final RMSE: {result['final_rmse']:.2f}K")
        print(f"  Improvement: {result['improvement_pct']:.1f}%")
        
        if result['improvement_pct'] > 5:
            print(f"  ✅ Active learning WORKS here!")
        else:
            print(f"  ❌ Minimal benefit")
    
    # Save results
    output_dir = Path("validation/conditions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Where Active Learning Works")
    print("="*70)
    
    best = max(results, key=lambda x: x['improvement_pct'])
    print(f"\n🏆 Best condition: {best['name']}")
    print(f"   Improvement: {best['improvement_pct']:.1f}%")
    print(f"   RMSE: {best['initial_rmse']:.2f}K → {best['final_rmse']:.2f}K")
    
    print("\n📊 All results ranked:")
    for r in sorted(results, key=lambda x: x['improvement_pct'], reverse=True):
        print(f"   {r['improvement_pct']:6.1f}% - {r['name']}")
    
    print("\n✅ Saved to: validation/conditions/results.json")

if __name__ == "__main__":
    main()

