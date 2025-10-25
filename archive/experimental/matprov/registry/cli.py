"""
CLI for prediction registry management
"""

import click
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from matprov.registry.database import Database
from matprov.registry.models import Model, Prediction, ExperimentOutcome, PredictionError
from matprov.registry.queries import PredictionQueries


@click.group()
def cli():
    """Prediction Registry: Track ML predictions and experimental validation"""
    pass


@cli.command()
@click.option('--db-url', default=None, help='Database URL (default: .matprov/predictions.db)')
def init(db_url: Optional[str]):
    """Initialize prediction registry database"""
    db = Database(db_url)
    db.create_tables()
    click.echo(f"✅ Initialized prediction registry: {db.url}")


@cli.command()
@click.option('--model-id', required=True, help='Model identifier')
@click.option('--version', required=True, help='Model version')
@click.option('--checkpoint', required=True, help='DVC hash of model checkpoint')
@click.option('--dataset', required=True, help='DVC hash of training dataset')
@click.option('--architecture', default=None, help='Model architecture')
@click.option('--notes', default=None, help='Additional notes')
def add_model(model_id: str, version: str, checkpoint: str, dataset: str, 
              architecture: Optional[str], notes: Optional[str]):
    """Register a new model"""
    db = Database()
    
    with db.session() as session:
        model = Model(
            model_id=model_id,
            version=version,
            checkpoint_hash=checkpoint,
            training_dataset_hash=dataset,
            architecture=architecture,
            notes=notes
        )
        session.add(model)
    
    click.echo(f"✅ Registered model: {model_id} v{version}")


@cli.command()
@click.argument('prediction_json', type=click.Path(exists=True, path_type=Path))
def add_prediction(prediction_json: Path):
    """Add prediction from JSON file"""
    with open(prediction_json) as f:
        data = json.load(f)
    
    db = Database()
    
    with db.session() as session:
        # Get model by model_id
        from sqlalchemy import select
        model = session.execute(
            select(Model).where(Model.model_id == data['model_id'])
        ).scalar_one()
        
        pred = Prediction(
            prediction_id=data['prediction_id'],
            model_id=model.id,
            material_formula=data['material_formula'],
            predicted_tc=data['predicted_tc'],
            uncertainty=data.get('uncertainty'),
            predicted_class=data.get('predicted_class'),
            confidence=data.get('confidence')
        )
        session.add(pred)
    
    click.echo(f"✅ Added prediction: {data['prediction_id']}")


@cli.command()
@click.option('--prediction-id', required=True, help='Prediction ID')
@click.option('--experiment-id', required=True, help='Experiment ID from matprov')
@click.option('--actual-tc', required=True, type=float, help='Measured Tc (K)')
@click.option('--status', default='success', help='Validation status')
@click.option('--purity', type=float, help='Phase purity (%)')
def add_outcome(prediction_id: str, experiment_id: str, actual_tc: float, 
                status: str, purity: Optional[float]):
    """Add experimental outcome for a prediction"""
    db = Database()
    
    with db.session() as session:
        # Get prediction
        from sqlalchemy import select
        pred = session.execute(
            select(Prediction).where(Prediction.prediction_id == prediction_id)
        ).scalar_one()
        
        # Add outcome
        outcome = ExperimentOutcome(
            prediction_id=pred.id,
            experiment_id=experiment_id,
            actual_tc=actual_tc,
            validation_status=status,
            phase_purity=purity,
            is_superconductor=actual_tc > 0
        )
        session.add(outcome)
        session.flush()
        
        # Compute error
        error = PredictionError(
            prediction_id=pred.id,
            absolute_error=pred.predicted_tc - actual_tc,
            relative_error=((pred.predicted_tc - actual_tc) / actual_tc * 100) if actual_tc != 0 else None,
            squared_error=(pred.predicted_tc - actual_tc) ** 2
        )
        session.add(error)
    
    click.echo(f"✅ Added outcome for {prediction_id}: Tc={actual_tc}K")
    click.echo(f"   Error: {pred.predicted_tc - actual_tc:.2f}K")


@cli.command()
@click.option('--threshold', default=10.0, type=float, help='Error threshold (K)')
@click.option('--limit', default=10, type=int, help='Maximum results')
def large_errors(threshold: float, limit: int):
    """Show predictions with large errors"""
    db = Database()
    
    with db.session() as session:
        queries = PredictionQueries(session)
        results = queries.predictions_with_large_errors(threshold, limit)
        
        if not results:
            click.echo(f"No predictions with |error| > {threshold}K")
            return
        
        click.echo(f"\n📊 Predictions with |error| > {threshold}K:\n")
        for r in results:
            click.echo(f"  {r['prediction_id']}: {r['material_formula']}")
            click.echo(f"    Predicted: {r['predicted_tc']:.1f}K, Actual: {r['actual_tc']:.1f}K")
            click.echo(f"    Error: {r['absolute_error']:.2f}K ({r['relative_error']:.1f}%)")
            click.echo(f"    Experiment: {r['experiment_id']} ({r['validation_status']})\n")


@cli.command()
@click.argument('model_id')
@click.option('--status', help='Filter by validation status')
def model_experiments(model_id: str, status: Optional[str]):
    """Show experiments validating a model"""
    db = Database()
    
    with db.session() as session:
        queries = PredictionQueries(session)
        results = queries.experiments_validating_model(model_id, status)
        
        if not results:
            click.echo(f"No experiments found for model {model_id}")
            return
        
        click.echo(f"\n🧪 Experiments validating {model_id}:\n")
        for r in results:
            click.echo(f"  {r['experiment_id']}: {r['material_formula']}")
            click.echo(f"    Predicted: {r['predicted_tc']:.1f}K → Actual: {r['actual_tc']:.1f}K")
            click.echo(f"    Status: {r['validation_status']}, Purity: {r['phase_purity']:.1f}%\n")


@cli.command()
@click.argument('model_id')
def performance(model_id: str):
    """Show model performance summary"""
    db = Database()
    
    with db.session() as session:
        queries = PredictionQueries(session)
        perf = queries.model_performance_summary(model_id)
        
        if 'error' in perf:
            click.echo(f"❌ {perf['error']}")
            return
        
        click.echo(f"\n📈 Performance Summary: {model_id} v{perf['version']}\n")
        click.echo(f"  Total predictions: {perf['total_predictions']}")
        click.echo(f"  Validated: {perf['validated_predictions']}")
        click.echo(f"\n  Regression Metrics:")
        click.echo(f"    RMSE: {perf['rmse']:.2f}K")
        click.echo(f"    MAE: {perf['mae']:.2f}K")
        click.echo(f"    R²: {perf['r2_score']:.4f}")
        
        if perf['classification_metrics']:
            cm = perf['classification_metrics']
            click.echo(f"\n  Classification Metrics:")
            click.echo(f"    Accuracy: {cm['accuracy']:.1%}")
            click.echo(f"    Precision: {cm['precision']:.1%}")
            click.echo(f"    Recall: {cm['recall']:.1%}")
            click.echo(f"    F1: {cm['f1']:.1%}")


@cli.command()
@click.option('--model-id', help='Filter by model')
@click.option('--limit', default=10, type=int, help='Maximum results')
def unvalidated(model_id: Optional[str], limit: int):
    """Show unvalidated predictions"""
    db = Database()
    
    with db.session() as session:
        queries = PredictionQueries(session)
        results = queries.unvalidated_predictions(model_id, limit)
        
        if not results:
            click.echo("No unvalidated predictions")
            return
        
        click.echo(f"\n⏳ Unvalidated predictions:\n")
        for r in results:
            click.echo(f"  {r['prediction_id']}: {r['material_formula']}")
            click.echo(f"    Predicted Tc: {r['predicted_tc']:.1f}K ± {r['uncertainty']:.1f}K")
            click.echo(f"    Model: {r['model_id']} v{r['model_version']}\n")


@cli.command()
@click.option('--limit', default=10, type=int, help='Number of candidates')
@click.option('--min-tc', default=30.0, type=float, help='Minimum Tc (K)')
def candidates(limit: int, min_tc: float):
    """Show top candidates for validation"""
    db = Database()
    
    with db.session() as session:
        queries = PredictionQueries(session)
        results = queries.top_candidates_for_validation(limit, min_tc)
        
        if not results:
            click.echo("No candidates found")
            return
        
        click.echo(f"\n🎯 Top {limit} candidates for validation (Tc ≥ {min_tc}K):\n")
        for i, r in enumerate(results, 1):
            click.echo(f"  {i}. {r['material_formula']}")
            click.echo(f"     Predicted Tc: {r['predicted_tc']:.1f}K ± {r['uncertainty']:.1f}K")
            click.echo(f"     Confidence: {r['confidence']:.1%}")
            click.echo(f"     ID: {r['prediction_id']}\n")


if __name__ == "__main__":
    cli()

