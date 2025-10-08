"""
Experiment Selection CLI: Use Shannon entropy to prioritize experiments

Uses the trained UCI model to select high-value experiments based on:
- Prediction uncertainty (Shannon entropy)
- Boundary cases (near thresholds)
- Chemistry diversity
"""

import click
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from matprov.selector import ExperimentSelector


@click.command()
@click.option('--model', 'model_path', required=True, type=click.Path(exists=True, path_type=Path),
              help='Path to trained model pickle file')
@click.option('--dataset', required=True, type=click.Path(exists=True, path_type=Path),
              help='Path to CSV dataset with features')
@click.option('--k', default=10, type=int, help='Number of experiments to select')
@click.option('--min-tc', default=30.0, type=float, help='Minimum predicted Tc (K)')
@click.option('--max-tc', type=float, help='Maximum predicted Tc (K)')
@click.option('--exclude', type=click.Path(exists=True, path_type=Path),
              help='JSON file with already-validated material IDs')
@click.option('--output', type=click.Path(path_type=Path),
              help='Output JSON file (default: print to stdout)')
@click.option('--verbose', is_flag=True, help='Show detailed scores')
def main(model_path: Path, dataset: Path, k: int, min_tc: float, max_tc: float,
         exclude: Path, output: Path, verbose: bool):
    """
    Select top-K experiments using Shannon entropy.
    
    Example:
        python scripts/select_experiments.py \\
            --model models/superconductor_classifier.pkl \\
            --dataset data/superconductors/processed/uci_train.csv \\
            --k 10 --min-tc 30.0 --verbose
    """
    click.echo(f"🔬 Experiment Selector\n")
    
    # Load model
    selector = ExperimentSelector(model_path)
    click.echo(f"✅ Loaded model: {model_path}")
    click.echo(f"   Classes: {selector.class_names}")
    click.echo(f"   Boundaries: {selector.tc_boundaries}K\n")
    
    # Load dataset
    df = pd.read_csv(dataset)
    click.echo(f"✅ Loaded dataset: {dataset}")
    click.echo(f"   Samples: {len(df)}\n")
    
    # Extract features and IDs
    # Assume dataset has columns: material_id, material_formula, features...
    if 'material_id' in df.columns:
        material_ids = df['material_id'].tolist()
    else:
        material_ids = [f"MAT-{i:06d}" for i in range(len(df))]
    
    if 'material_formula' in df.columns:
        material_formulas = df['material_formula'].tolist()
    elif 'formula' in df.columns:
        material_formulas = df['formula'].tolist()
    else:
        material_formulas = [f"Material{i}" for i in range(len(df))]
    
    # Get feature columns (exclude metadata)
    exclude_cols = {'material_id', 'material_formula', 'formula', 'Tc', 'critical_temp', 'tc_class'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        click.echo("❌ No feature columns found in dataset", err=True)
        sys.exit(1)
    
    X = df[feature_cols].values
    click.echo(f"✅ Extracted features: {len(feature_cols)} columns\n")
    
    # Load exclusions
    exclude_ids = []
    if exclude and exclude.exists():
        with open(exclude) as f:
            exclude_data = json.load(f)
            exclude_ids = exclude_data.get('validated_ids', [])
        click.echo(f"✅ Loaded {len(exclude_ids)} validated materials to exclude\n")
    
    # Select experiments
    click.echo(f"🎯 Selecting top {k} experiments (Tc ≥ {min_tc}K)...\n")
    
    candidates = selector.select_experiments(
        X=X,
        material_ids=material_ids,
        material_formulas=material_formulas,
        k=k,
        min_tc=min_tc,
        max_tc=max_tc,
        exclude_ids=exclude_ids
    )
    
    if not candidates:
        click.echo("⚠️  No candidates found matching criteria")
        return
    
    click.echo(f"Selected {len(candidates)} candidates:\n")
    
    # Display results
    for i, c in enumerate(candidates, 1):
        click.echo(f"{i}. {c.material_formula} (ID: {c.material_id})")
        click.echo(f"   Predicted Tc: {c.predicted_tc:.1f}K")
        click.echo(f"   Entropy: {c.entropy:.3f} bits")
        
        if verbose:
            click.echo(f"   Predicted probs: {c.predicted_probs}")
            click.echo(f"   Scores: U={c.uncertainty_score:.3f}, B={c.boundary_score:.3f}, D={c.diversity_score:.3f}")
        
        click.echo(f"   Total Score: {c.total_score:.3f}\n")
    
    # Expected information gain
    eig = selector.expected_information_gain(candidates)
    click.echo(f"📊 Expected Information Gain: {eig:.2f} bits")
    click.echo(f"   ({eig/len(candidates):.3f} bits per experiment)\n")
    
    # Save to file
    if output:
        results = {
            'model_path': str(model_path),
            'dataset_path': str(dataset),
            'selection_criteria': {
                'k': k,
                'min_tc': min_tc,
                'max_tc': max_tc,
                'excluded_count': len(exclude_ids)
            },
            'candidates': [c.to_dict() for c in candidates],
            'expected_information_gain_bits': round(eig, 4),
            'bits_per_experiment': round(eig / len(candidates), 4)
        }
        
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"💾 Saved results to: {output}")


if __name__ == "__main__":
    main()

