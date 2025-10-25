"""
End-to-End Demo: UCI Superconductors → Predictions → Tracking → Validation

Demonstrates the complete workflow:
1. Load UCI dataset (21K superconductors)
2. Load trained model (88.8% accuracy)
3. Generate predictions with uncertainty
4. Track predictions in registry
5. Select experiments via Shannon entropy
6. Track experiments in matprov (with provenance)
7. Simulate validation outcomes
8. Compute prediction errors
9. Show provenance chain & performance

Runtime: < 5 minutes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from matprov.selector import ExperimentSelector
from matprov.registry.database import Database
from matprov.registry.models import Model, Prediction, ExperimentOutcome, PredictionError
from matprov.registry.queries import PredictionQueries
from matprov.schema import (
    MaterialsExperiment,
    ExperimentMetadata,
    SynthesisParameters,
    Precursor,
    TemperatureProfile,
    CharacterizationData,
    PredictionLink,
    Outcome,
    OutcomeStatus,
    EquipmentType
)


class EndToEndDemo:
    """Complete workflow demonstration"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.model_path = Path("models/superconductor_classifier.pkl")
        self.dataset_path = Path("data/superconductors/processed/train.csv")
        self.db_path = self.base_dir / "demo.db"
        
        # Initialize database
        self.db = Database(f"sqlite:///{self.db_path}")
        self.db.create_tables()
        
        print(f"🔬 End-to-End Demo Initialized")
        print(f"   Base dir: {self.base_dir}")
        print(f"   Database: {self.db_path}\n")
    
    def step1_load_data(self):
        """Step 1: Load UCI dataset"""
        print("=" * 70)
        print("STEP 1: Load UCI Superconductor Dataset")
        print("=" * 70 + "\n")
        
        if not self.dataset_path.exists():
            print(f"❌ Dataset not found: {self.dataset_path}")
            print("   Expected: data/superconductors/processed/train.csv")
            print("   Please run: python models/superconductor_classifier.py first\n")
            return False
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"✅ Loaded dataset: {len(self.df)} samples")
        print(f"   Features: {self.df.shape[1] - 1} columns")  # Exclude Tc
        print(f"   Tc range: {self.df['critical_temp'].min():.1f}K - {self.df['critical_temp'].max():.1f}K\n")
        
        return True
    
    def step2_load_model(self):
        """Step 2: Load trained model"""
        print("=" * 70)
        print("STEP 2: Load Trained RandomForest Model")
        print("=" * 70 + "\n")
        
        if not self.model_path.exists():
            print(f"❌ Model not found: {self.model_path}")
            print("   Please run: python models/superconductor_classifier.py first\n")
            return False
        
        self.selector = ExperimentSelector(self.model_path)
        print(f"✅ Loaded model: {self.model_path}")
        print(f"   Classes: {self.selector.class_names}")
        print(f"   Architecture: RandomForestClassifier\n")
        
        return True
    
    def step3_generate_predictions(self, n_samples: int = 100):
        """Step 3: Generate predictions"""
        print("=" * 70)
        print(f"STEP 3: Generate Predictions ({n_samples} samples)")
        print("=" * 70 + "\n")
        
        # Sample from dataset
        sample_df = self.df.sample(n=n_samples, random_state=42)
        
        # Extract features
        feature_cols = [col for col in sample_df.columns if col not in ['critical_temp', 'tc_class']]
        X = sample_df[feature_cols].values
        
        # Predict
        classes, probs, entropies = self.selector.predict_with_uncertainty(X)
        
        # Store results
        self.predictions_data = []
        tc_map = {0: 10.0, 1: 50.0, 2: 100.0}  # Class to Tc estimate
        
        for i, (idx, row) in enumerate(sample_df.iterrows()):
            pred_data = {
                'sample_id': f"SAMPLE-{idx:06d}",
                'actual_tc': float(row['critical_temp']),
                'predicted_class': int(classes[i]),
                'predicted_tc': tc_map[classes[i]],
                'predicted_probs': {
                    name: float(probs[i][j])
                    for j, name in enumerate(self.selector.class_names)
                },
                'entropy': float(entropies[i]),
                'features': X[i]
            }
            self.predictions_data.append(pred_data)
        
        print(f"✅ Generated {len(self.predictions_data)} predictions")
        print(f"   Avg entropy: {np.mean(entropies):.3f} bits")
        print(f"   Max entropy: {np.max(entropies):.3f} bits")
        print(f"   Min entropy: {np.min(entropies):.3f} bits\n")
        
        return True
    
    def step4_track_predictions(self):
        """Step 4: Track predictions in registry"""
        print("=" * 70)
        print("STEP 4: Track Predictions in Registry Database")
        print("=" * 70 + "\n")
        
        with self.db.session() as session:
            # Register model
            model = Model(
                model_id="rf_v2.1_uci_21k_demo",
                version="2.1.0",
                checkpoint_hash="dvc:demo_checkpoint",
                training_dataset_hash="dvc:3f34e6c71b4245aad0da5acc3d39fe7f",
                architecture="RandomForestClassifier",
                hyperparameters='{"n_estimators": 100, "max_depth": 20}',
                notes="Trained on UCI Superconductor dataset (21,263 samples) - DEMO"
            )
            session.add(model)
            session.flush()
            
            # Add predictions
            for pred_data in self.predictions_data:
                pred = Prediction(
                    prediction_id=pred_data['sample_id'],
                    model_id=model.id,
                    material_formula=f"Material-{pred_data['sample_id']}",
                    predicted_tc=pred_data['predicted_tc'],
                    uncertainty=pred_data['entropy'] * 10.0,  # Convert bits to K estimate
                    predicted_class=self.selector.class_names[pred_data['predicted_class']],
                    confidence=1.0 - pred_data['entropy'] / math.log(3, 2)  # Normalize
                )
                session.add(pred)
                pred_data['db_pred_id'] = pred.id
        
        print(f"✅ Tracked {len(self.predictions_data)} predictions")
        print(f"   Model ID: {model.model_id}")
        print(f"   Database: {self.db_path}\n")
        
        return True
    
    def step5_select_experiments(self, k: int = 10):
        """Step 5: Select experiments via Shannon entropy"""
        print("=" * 70)
        print(f"STEP 5: Select Top-{k} Experiments (Shannon Entropy)")
        print("=" * 70 + "\n")
        
        # Prepare data
        X = np.array([p['features'] for p in self.predictions_data])
        material_ids = [p['sample_id'] for p in self.predictions_data]
        material_formulas = [f"Material-{p['sample_id']}" for p in self.predictions_data]
        
        # Select
        candidates = self.selector.select_experiments(
            X=X,
            material_ids=material_ids,
            material_formulas=material_formulas,
            k=k,
            min_tc=0.0  # No filtering for demo
        )
        
        self.selected_candidates = candidates
        
        print(f"✅ Selected {len(candidates)} candidates")
        print(f"\n   Top 5:")
        for i, c in enumerate(candidates[:5], 1):
            print(f"   {i}. {c.material_id}")
            print(f"      Predicted Tc: {c.predicted_tc:.1f}K, Entropy: {c.entropy:.3f} bits")
            print(f"      Score: {c.total_score:.3f}\n")
        
        # Expected information gain
        eig = self.selector.expected_information_gain(candidates)
        print(f"📊 Expected Information Gain: {eig:.2f} bits")
        print(f"   ({eig/len(candidates):.3f} bits per experiment)\n")
        
        return True
    
    def step6_track_experiments(self):
        """Step 6: Track experiments in matprov"""
        print("=" * 70)
        print("STEP 6: Track Experiments in matprov (Provenance)")
        print("=" * 70 + "\n")
        
        experiments_dir = self.base_dir / "experiments"
        experiments_dir.mkdir(exist_ok=True)
        
        self.experiment_records = []
        
        for i, candidate in enumerate(self.selected_candidates[:5], 1):  # Track first 5 for demo
            # Find prediction data
            pred_data = next(p for p in self.predictions_data if p['sample_id'] == candidate.material_id)
            
            # Create experiment record
            exp_id = f"DEMO-EXP-{i:03d}"
            
            experiment = MaterialsExperiment(
                metadata=ExperimentMetadata(
                    experiment_id=exp_id,
                    operator="demo@periodic.labs",
                    equipment_ids={
                        EquipmentType.FURNACE: "DEMO-FURNACE-1"
                    },
                    project_id="UCI-DEMO"
                ),
                synthesis=SynthesisParameters(
                    precursors=[
                        Precursor(
                            chemical_formula="DemoChemical",
                            batch_number=f"BATCH-{i}",
                            supplier="Demo Supplier",
                            purity=0.99,
                            amount_g=1.0
                        )
                    ],
                    target_formula=candidate.material_formula,
                    temperature_profile=TemperatureProfile(
                        ramp_rate_c_per_min=5.0,
                        hold_temperature_c=900.0,
                        hold_duration_min=120.0
                    ),
                    atmosphere="air"
                ),
                characterization=CharacterizationData(),
                prediction=PredictionLink(
                    model_id="rf_v2.1_uci_21k_demo",
                    model_version="2.1.0",
                    predicted_properties={"Tc": candidate.predicted_tc},
                    confidence_scores={"entropy": candidate.entropy},
                    training_data_hash="dvc:3f34e6c71b4245aad0da5acc3d39fe7f"
                ),
                outcome=Outcome(
                    status=OutcomeStatus.SUCCESS,
                    actual_properties={"Tc": pred_data['actual_tc']},
                    notes=f"Demo experiment - Actual Tc from UCI dataset"
                )
            )
            
            # Save to file
            exp_path = experiments_dir / f"{exp_id}.json"
            experiment.export_json(exp_path)
            
            self.experiment_records.append({
                'experiment_id': exp_id,
                'experiment': experiment,
                'path': exp_path,
                'pred_data': pred_data
            })
        
        print(f"✅ Created {len(self.experiment_records)} experiment records")
        print(f"   Saved to: {experiments_dir}/\n")
        
        return True
    
    def step7_validate_outcomes(self):
        """Step 7: Add experimental outcomes to registry"""
        print("=" * 70)
        print("STEP 7: Add Experimental Outcomes & Compute Errors")
        print("=" * 70 + "\n")
        
        with self.db.session() as session:
            queries = PredictionQueries(session)
            
            for exp_rec in self.experiment_records:
                pred_data = exp_rec['pred_data']
                exp_id = exp_rec['experiment_id']
                
                # Find prediction in DB
                from sqlalchemy import select
                pred = session.execute(
                    select(Prediction).where(Prediction.prediction_id == pred_data['sample_id'])
                ).scalar_one()
                
                # Add outcome
                outcome = ExperimentOutcome(
                    prediction_id=pred.id,
                    experiment_id=exp_id,
                    actual_tc=pred_data['actual_tc'],
                    validation_status="success",
                    is_superconductor=pred_data['actual_tc'] > 0
                )
                session.add(outcome)
                session.flush()
                
                # Compute error
                error = PredictionError(
                    prediction_id=pred.id,
                    absolute_error=pred.predicted_tc - pred_data['actual_tc'],
                    relative_error=((pred.predicted_tc - pred_data['actual_tc']) / pred_data['actual_tc'] * 100) 
                        if pred_data['actual_tc'] != 0 else None,
                    squared_error=(pred.predicted_tc - pred_data['actual_tc']) ** 2
                )
                session.add(error)
                
                print(f"✅ {exp_id}: Predicted={pred.predicted_tc:.1f}K, Actual={pred_data['actual_tc']:.1f}K")
                print(f"   Error: {error.absolute_error:.2f}K\n")
        
        return True
    
    def step8_show_provenance(self):
        """Step 8: Show provenance & performance"""
        print("=" * 70)
        print("STEP 8: Provenance Chain & Performance Summary")
        print("=" * 70 + "\n")
        
        # Show experiment provenance
        print("📋 Experiment Provenance:\n")
        for exp_rec in self.experiment_records[:3]:  # First 3
            exp = exp_rec['experiment']
            print(f"   {exp.metadata.experiment_id}:")
            print(f"   ├─ Target: {exp.synthesis.target_formula}")
            print(f"   ├─ Predicted Tc: {exp.prediction.predicted_properties['Tc']:.1f}K")
            print(f"   ├─ Actual Tc: {exp.outcome.actual_properties['Tc']:.1f}K")
            print(f"   ├─ Content Hash: {exp.content_hash()[:16]}...")
            print(f"   └─ Status: {exp.outcome.status.value}\n")
        
        # Model performance
        with self.db.session() as session:
            queries = PredictionQueries(session)
            perf = queries.model_performance_summary("rf_v2.1_uci_21k_demo")
            
            if 'error' not in perf:
                print("📈 Model Performance:\n")
                print(f"   Validated Predictions: {perf['validated_predictions']}")
                print(f"   RMSE: {perf['rmse']:.2f}K")
                print(f"   MAE: {perf['mae']:.2f}K")
                print(f"   R²: {perf['r2_score']:.4f}\n")
        
        return True
    
    def run_full_demo(self):
        """Run complete demo"""
        import math  # Import here for step methods
        
        print("\n" + "=" * 70)
        print("🔬 END-TO-END DEMO: Materials Discovery with Provenance")
        print("=" * 70 + "\n")
        
        steps = [
            (self.step1_load_data, "Load UCI Dataset"),
            (self.step2_load_model, "Load Trained Model"),
            (self.step3_generate_predictions, "Generate Predictions"),
            (self.step4_track_predictions, "Track in Registry"),
            (self.step5_select_experiments, "Select Experiments"),
            (self.step6_track_experiments, "Track with Provenance"),
            (self.step7_validate_outcomes, "Validate Outcomes"),
            (self.step8_show_provenance, "Show Provenance")
        ]
        
        for step_func, step_name in steps:
            if not step_func():
                print(f"❌ Demo stopped at: {step_name}")
                return False
        
        print("=" * 70)
        print("✅ END-TO-END DEMO COMPLETE!")
        print("=" * 70)
        print(f"\nArtifacts:")
        print(f"   Database: {self.db_path}")
        print(f"   Experiments: {self.base_dir / 'experiments'}/")
        print(f"\nNext steps:")
        print(f"   - Query database: python -m matprov.registry.cli performance rf_v2.1_uci_21k_demo")
        print(f"   - Verify experiments: matprov verify <exp_id>")
        print(f"   - View lineage: matprov lineage <exp_id>\n")
        
        return True


if __name__ == "__main__":
    import math  # For step methods
    
    demo_dir = Path("demo/output")
    demo = EndToEndDemo(demo_dir)
    
    success = demo.run_full_demo()
    sys.exit(0 if success else 1)

