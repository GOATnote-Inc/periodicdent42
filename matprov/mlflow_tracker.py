"""
MLflow Integration for matprov

Track model training, predictions, and retraining with full provenance.

Features:
- Log model training with hyperparameters
- Track dataset lineage (DVC hashes)
- Log predictions and outcomes
- Auto-link to matprov registry
- Retrain triggers with data drift detection
"""

import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import json
from datetime import datetime


class MatprovMLflowTracker:
    """MLflow tracker with matprov integration"""
    
    def __init__(self, experiment_name: str = "matprov_superconductors", tracking_uri: Optional[str] = None):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI (default: local ./mlruns)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
    
    def log_training_run(
        self,
        model: Any,
        dataset_hash: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        model_name: str = "superconductor_classifier",
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Log a training run with full provenance
        
        Args:
            model: Trained scikit-learn model
            dataset_hash: DVC hash of training dataset
            hyperparameters: Model hyperparameters
            metrics: Training metrics (accuracy, f1, etc.)
            model_name: Name for the model
            tags: Additional tags
        
        Returns:
            MLflow run ID
        """
        with mlflow.start_run() as run:
            # Log hyperparameters
            mlflow.log_params(hyperparameters)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log dataset provenance
            mlflow.log_param("dataset_hash", dataset_hash)
            mlflow.log_param("dataset_source", "dvc")
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                model_name,
                registered_model_name=model_name
            )
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Log matprov metadata
            mlflow.set_tag("matprov.tracked", "true")
            mlflow.set_tag("matprov.timestamp", datetime.now().isoformat())
            
            # Compute model hash
            model_hash = self._compute_model_hash(model)
            mlflow.log_param("model_hash", model_hash)
            
            run_id = run.info.run_id
            
            print(f"✅ MLflow run logged: {run_id}")
            print(f"   Dataset: {dataset_hash[:16]}...")
            print(f"   Model: {model_hash[:16]}...")
            
            return run_id
    
    def log_prediction_batch(
        self,
        run_id: str,
        predictions: List[Dict[str, Any]]
    ):
        """
        Log a batch of predictions
        
        Args:
            run_id: MLflow run ID
            predictions: List of prediction dictionaries
        """
        with mlflow.start_run(run_id=run_id):
            # Log prediction count
            mlflow.log_metric("num_predictions", len(predictions))
            
            # Log prediction statistics
            predicted_tcs = [p['predicted_tc'] for p in predictions]
            mlflow.log_metric("mean_predicted_tc", sum(predicted_tcs) / len(predicted_tcs))
            mlflow.log_metric("max_predicted_tc", max(predicted_tcs))
            mlflow.log_metric("min_predicted_tc", min(predicted_tcs))
            
            # Save predictions as artifact
            predictions_file = Path("predictions.json")
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            mlflow.log_artifact(str(predictions_file))
            predictions_file.unlink()
            
            print(f"✅ Logged {len(predictions)} predictions to run {run_id}")
    
    def log_validation_results(
        self,
        run_id: str,
        outcomes: List[Dict[str, Any]]
    ):
        """
        Log experimental validation results
        
        Args:
            run_id: MLflow run ID
            outcomes: List of outcome dictionaries
        """
        with mlflow.start_run(run_id=run_id):
            # Compute validation metrics
            errors = [o['predicted_tc'] - o['actual_tc'] for o in outcomes]
            mae = sum(abs(e) for e in errors) / len(errors)
            rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
            
            mlflow.log_metric("validation_mae", mae)
            mlflow.log_metric("validation_rmse", rmse)
            mlflow.log_metric("num_validated", len(outcomes))
            
            # Save outcomes as artifact
            outcomes_file = Path("validation_outcomes.json")
            with open(outcomes_file, 'w') as f:
                json.dump(outcomes, f, indent=2)
            
            mlflow.log_artifact(str(outcomes_file))
            outcomes_file.unlink()
            
            print(f"✅ Logged {len(outcomes)} validation outcomes")
            print(f"   MAE: {mae:.2f}K")
            print(f"   RMSE: {rmse:.2f}K")
    
    def should_retrain(
        self,
        current_run_id: str,
        new_data_count: int,
        retrain_threshold: int = 50,
        performance_drop_threshold: float = 0.1
    ) -> bool:
        """
        Determine if model should be retrained
        
        Args:
            current_run_id: Current MLflow run ID
            new_data_count: Number of new validated experiments
            retrain_threshold: Minimum new data points to trigger retrain
            performance_drop_threshold: Max acceptable performance drop (fraction)
        
        Returns:
            True if should retrain
        """
        # Check data count
        if new_data_count >= retrain_threshold:
            print(f"✅ Retrain triggered: {new_data_count} >= {retrain_threshold} new samples")
            return True
        
        # Check performance drop
        run = mlflow.get_run(current_run_id)
        if 'validation_mae' in run.data.metrics:
            training_mae = run.data.metrics.get('mae', 0)
            validation_mae = run.data.metrics.get('validation_mae', 0)
            
            if validation_mae > training_mae * (1 + performance_drop_threshold):
                print(f"✅ Retrain triggered: Performance drop ({validation_mae:.2f}K > {training_mae * (1 + performance_drop_threshold):.2f}K)")
                return True
        
        print(f"⏳ No retrain needed yet ({new_data_count}/{retrain_threshold} new samples)")
        return False
    
    def get_model_lineage(self, run_id: str) -> Dict[str, Any]:
        """
        Get full lineage for a model
        
        Args:
            run_id: MLflow run ID
        
        Returns:
            Dictionary with lineage information
        """
        run = mlflow.get_run(run_id)
        
        lineage = {
            'run_id': run_id,
            'experiment_id': run.info.experiment_id,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags,
            'dataset_hash': run.data.params.get('dataset_hash'),
            'model_hash': run.data.params.get('model_hash'),
            'artifacts': mlflow.artifacts.list_artifacts(run_id)
        }
        
        return lineage
    
    @staticmethod
    def _compute_model_hash(model: Any) -> str:
        """Compute hash of model parameters"""
        import pickle
        model_bytes = pickle.dumps(model)
        return hashlib.sha256(model_bytes).hexdigest()


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    
    print("=== MLflow Integration Demo ===\n")
    
    # Create tracker
    tracker = MatprovMLflowTracker(experiment_name="demo_experiment")
    print("✅ Initialized MLflow tracker")
    
    # Generate dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Compute metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"✅ Trained model: accuracy={accuracy:.3f}, f1={f1:.3f}")
    
    # Log training run
    run_id = tracker.log_training_run(
        model=model,
        dataset_hash="dvc:3f34e6c71b4245aad0da5acc3d39fe7f",
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        metrics={
            'accuracy': accuracy,
            'f1_score': f1
        },
        model_name="demo_classifier",
        tags={'version': '1.0', 'framework': 'sklearn'}
    )
    
    # Log predictions
    predictions = [
        {'material_id': 'MAT-001', 'predicted_tc': 92.5},
        {'material_id': 'MAT-002', 'predicted_tc': 45.3},
        {'material_id': 'MAT-003', 'predicted_tc': 12.1},
    ]
    tracker.log_prediction_batch(run_id, predictions)
    
    # Log validation
    outcomes = [
        {'material_id': 'MAT-001', 'predicted_tc': 92.5, 'actual_tc': 89.3},
        {'material_id': 'MAT-002', 'predicted_tc': 45.3, 'actual_tc': 47.1},
    ]
    tracker.log_validation_results(run_id, outcomes)
    
    # Check retrain
    should_retrain = tracker.should_retrain(run_id, new_data_count=30)
    
    # Get lineage
    lineage = tracker.get_model_lineage(run_id)
    print(f"\n📊 Model Lineage:")
    print(f"   Run ID: {lineage['run_id']}")
    print(f"   Dataset: {lineage['dataset_hash']}")
    print(f"   Model Hash: {lineage['model_hash'][:16]}...")
    
    print("\n✅ MLflow integration complete!")
    print(f"\nView results: mlflow ui")

