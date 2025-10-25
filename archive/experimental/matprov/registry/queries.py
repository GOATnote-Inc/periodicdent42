"""
Query interface for prediction registry

Provides high-level query methods for analyzing predictions and outcomes.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import Session, joinedload

from matprov.registry.models import Model, Prediction, ExperimentOutcome, PredictionError


class PredictionQueries:
    """
    High-level query interface for prediction registry.
    
    Examples:
        - Show me all predictions with |error| > 10K
        - Which experiments validated model v2.3?
        - Calculate true positive rate for Tc > 30K predictions
    """
    
    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session
    
    def predictions_with_large_errors(
        self,
        threshold: float = 10.0,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all predictions with |absolute_error| > threshold.
        
        Args:
            threshold: Error threshold in Kelvin
            limit: Maximum number of results
            
        Returns:
            List of dicts with prediction and error information
        """
        query = (
            select(Prediction, PredictionError, ExperimentOutcome)
            .join(PredictionError, Prediction.id == PredictionError.prediction_id)
            .join(ExperimentOutcome, Prediction.id == ExperimentOutcome.prediction_id)
            .where(func.abs(PredictionError.absolute_error) > threshold)
            .order_by(func.abs(PredictionError.absolute_error).desc())
        )
        
        if limit:
            query = query.limit(limit)
        
        results = []
        for pred, error, outcome in self.session.execute(query):
            results.append({
                'prediction_id': pred.prediction_id,
                'material_formula': pred.material_formula,
                'predicted_tc': pred.predicted_tc,
                'actual_tc': outcome.actual_tc,
                'absolute_error': error.absolute_error,
                'relative_error': error.relative_error,
                'experiment_id': outcome.experiment_id,
                'validation_status': outcome.validation_status
            })
        
        return results
    
    def experiments_validating_model(
        self,
        model_id: str,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all experiments that validated a specific model.
        
        Args:
            model_id: Model identifier (e.g., "rf_v2.3_uci")
            status: Filter by validation status (e.g., "success")
            
        Returns:
            List of dicts with experiment information
        """
        query = (
            select(Prediction, ExperimentOutcome, Model)
            .join(ExperimentOutcome, Prediction.id == ExperimentOutcome.prediction_id)
            .join(Model, Prediction.model_id == Model.id)
            .where(Model.model_id == model_id)
        )
        
        if status:
            query = query.where(ExperimentOutcome.validation_status == status)
        
        results = []
        for pred, outcome, model in self.session.execute(query):
            results.append({
                'experiment_id': outcome.experiment_id,
                'material_formula': pred.material_formula,
                'predicted_tc': pred.predicted_tc,
                'actual_tc': outcome.actual_tc,
                'validation_status': outcome.validation_status,
                'phase_purity': outcome.phase_purity,
                'model_version': model.version,
                'experiment_date': outcome.experiment_date
            })
        
        return results
    
    def calculate_true_positive_rate(
        self,
        threshold: float = 30.0,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate classification metrics for Tc > threshold predictions.
        
        Args:
            threshold: Temperature threshold in Kelvin
            model_id: Optional model filter
            
        Returns:
            Dict with TPR, FPR, precision, recall, F1
        """
        query = select(PredictionError)
        
        if model_id:
            query = (
                query.join(Prediction, PredictionError.prediction_id == Prediction.id)
                .join(Model, Prediction.model_id == Model.id)
                .where(Model.model_id == model_id)
            )
        
        errors = self.session.execute(query).scalars().all()
        
        if not errors:
            return {'error': 'No data available'}
        
        tp = sum(1 for e in errors if e.is_true_positive)
        tn = sum(1 for e in errors if e.is_true_negative)
        fp = sum(1 for e in errors if e.is_false_positive)
        fn = sum(1 for e in errors if e.is_false_negative)
        
        total = tp + tn + fp + fn
        
        # Handle division by zero
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'threshold': threshold,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'total': total,
            'true_positive_rate': round(tpr, 4),
            'false_positive_rate': round(fpr, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'accuracy': round((tp + tn) / total, 4) if total > 0 else 0.0
        }
    
    def model_performance_summary(
        self,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dict with RMSE, MAE, R^2, classification metrics
        """
        # Get model
        model = self.session.execute(
            select(Model).where(Model.model_id == model_id)
        ).scalar_one_or_none()
        
        if not model:
            return {'error': f'Model {model_id} not found'}
        
        # Get all predictions with outcomes
        query = (
            select(Prediction, PredictionError, ExperimentOutcome)
            .join(PredictionError, Prediction.id == PredictionError.prediction_id)
            .join(ExperimentOutcome, Prediction.id == ExperimentOutcome.prediction_id)
            .where(Prediction.model_id == model.id)
        )
        
        results = list(self.session.execute(query))
        
        if not results:
            return {'error': f'No validated predictions for model {model_id}'}
        
        # Compute regression metrics
        absolute_errors = [error.absolute_error for _, error, _ in results]
        squared_errors = [error.squared_error for _, error, _ in results]
        
        mae = sum(abs(e) for e in absolute_errors) / len(absolute_errors)
        rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
        
        # Compute R^2
        actual_values = [outcome.actual_tc for _, _, outcome in results]
        predicted_values = [pred.predicted_tc for pred, _, _ in results]
        mean_actual = sum(actual_values) / len(actual_values)
        
        ss_tot = sum((y - mean_actual) ** 2 for y in actual_values)
        ss_res = sum(squared_errors)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Classification metrics (if available)
        errors_with_class = [error for _, error, _ in results if error.is_true_positive is not None]
        
        classification_metrics = None
        if errors_with_class:
            tp = sum(1 for e in errors_with_class if e.is_true_positive)
            tn = sum(1 for e in errors_with_class if e.is_true_negative)
            fp = sum(1 for e in errors_with_class if e.is_false_positive)
            fn = sum(1 for e in errors_with_class if e.is_false_negative)
            total = tp + tn + fp + fn
            
            classification_metrics = {
                'accuracy': round((tp + tn) / total, 4) if total > 0 else 0.0,
                'precision': round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0,
                'recall': round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
                'f1': round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) > 0 else 0.0
            }
        
        return {
            'model_id': model_id,
            'version': model.version,
            'total_predictions': len(results),
            'validated_predictions': len(results),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'r2_score': round(r2, 4),
            'mean_actual_tc': round(mean_actual, 2),
            'classification_metrics': classification_metrics
        }
    
    def unvalidated_predictions(
        self,
        model_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get predictions that haven't been experimentally validated yet.
        
        Args:
            model_id: Optional model filter
            limit: Maximum number of results
            
        Returns:
            List of unvalidated predictions
        """
        # Subquery for predictions with outcomes
        validated_ids = select(ExperimentOutcome.prediction_id)
        
        query = (
            select(Prediction, Model)
            .join(Model, Prediction.model_id == Model.id)
            .where(Prediction.id.not_in(validated_ids))
        )
        
        if model_id:
            query = query.where(Model.model_id == model_id)
        
        query = query.order_by(Prediction.prediction_date.desc())
        
        if limit:
            query = query.limit(limit)
        
        results = []
        for pred, model in self.session.execute(query):
            results.append({
                'prediction_id': pred.prediction_id,
                'material_formula': pred.material_formula,
                'predicted_tc': pred.predicted_tc,
                'uncertainty': pred.uncertainty,
                'confidence': pred.confidence,
                'model_id': model.model_id,
                'model_version': model.version,
                'prediction_date': pred.prediction_date
            })
        
        return results
    
    def top_candidates_for_validation(
        self,
        limit: int = 10,
        min_tc: float = 30.0,
        max_uncertainty: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top candidate predictions for experimental validation.
        
        Prioritizes:
        - High predicted Tc
        - Reasonable uncertainty
        - Not yet validated
        
        Args:
            limit: Number of candidates to return
            min_tc: Minimum predicted Tc threshold
            max_uncertainty: Maximum acceptable uncertainty
            
        Returns:
            List of top candidates
        """
        validated_ids = select(ExperimentOutcome.prediction_id)
        
        query = (
            select(Prediction, Model)
            .join(Model, Prediction.model_id == Model.id)
            .where(
                and_(
                    Prediction.id.not_in(validated_ids),
                    Prediction.predicted_tc >= min_tc
                )
            )
        )
        
        if max_uncertainty is not None:
            query = query.where(Prediction.uncertainty <= max_uncertainty)
        
        query = query.order_by(Prediction.predicted_tc.desc()).limit(limit)
        
        results = []
        for pred, model in self.session.execute(query):
            results.append({
                'prediction_id': pred.prediction_id,
                'material_formula': pred.material_formula,
                'predicted_tc': pred.predicted_tc,
                'uncertainty': pred.uncertainty,
                'confidence': pred.confidence,
                'model_id': model.model_id,
                'model_version': model.version
            })
        
        return results


# Example usage
if __name__ == "__main__":
    from matprov.registry.database import Database
    from matprov.registry.models import Model, Prediction, ExperimentOutcome, PredictionError
    
    # Create in-memory database for testing
    db = Database("sqlite:///:memory:")
    db.create_tables()
    
    with db.session() as session:
        # Create test data
        model = Model(
            model_id="rf_v2.3_uci",
            version="2.3.0",
            checkpoint_hash="test_checkpoint",
            training_dataset_hash="test_dataset"
        )
        session.add(model)
        session.flush()
        
        # Add predictions with outcomes
        for i in range(5):
            pred = Prediction(
                prediction_id=f"PRED-{i:03d}",
                model_id=model.id,
                material_formula=f"Material{i}",
                predicted_tc=50.0 + i * 10,
                uncertainty=5.0
            )
            session.add(pred)
            session.flush()
            
            outcome = ExperimentOutcome(
                prediction_id=pred.id,
                experiment_id=f"EXP-{i:03d}",
                actual_tc=pred.predicted_tc + (i - 2) * 5,  # Vary error
                validation_status="success",
                is_superconductor=True
            )
            session.add(outcome)
            session.flush()
            
            error = PredictionError(
                prediction_id=pred.id,
                absolute_error=pred.predicted_tc - outcome.actual_tc,
                squared_error=(pred.predicted_tc - outcome.actual_tc) ** 2,
                is_true_positive=True if pred.predicted_tc > 30 else False
            )
            session.add(error)
    
    # Test queries
    with db.session() as session:
        queries = PredictionQueries(session)
        
        print("=== Test Queries ===\n")
        
        # Query 1: Large errors
        large_errors = queries.predictions_with_large_errors(threshold=7.0)
        print(f"1. Predictions with |error| > 7K: {len(large_errors)}")
        for pred in large_errors:
            print(f"   - {pred['material_formula']}: error={pred['absolute_error']:.2f}K")
        
        # Query 2: Experiments validating model
        exps = queries.experiments_validating_model("rf_v2.3_uci")
        print(f"\n2. Experiments validating rf_v2.3_uci: {len(exps)}")
        
        # Query 3: TPR
        metrics = queries.calculate_true_positive_rate(threshold=30.0)
        print(f"\n3. Classification metrics (Tc > 30K):")
        print(f"   TPR: {metrics['true_positive_rate']}")
        print(f"   FPR: {metrics['false_positive_rate']}")
        print(f"   F1: {metrics['f1_score']}")
        
        # Query 4: Model performance
        perf = queries.model_performance_summary("rf_v2.3_uci")
        print(f"\n4. Model performance summary:")
        print(f"   RMSE: {perf['rmse']:.2f}K")
        print(f"   MAE: {perf['mae']:.2f}K")
        print(f"   R²: {perf['r2_score']:.4f}")
        
        print("\n✅ All queries executed successfully!")

