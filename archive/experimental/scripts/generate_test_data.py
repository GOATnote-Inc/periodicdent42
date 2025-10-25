#!/usr/bin/env python3
"""
Generate realistic test data for Cloud SQL database.

This script populates the database with:
- Experiments with varying parameters and noise levels
- Optimization runs using different methods (RL, BO, Adaptive)
- AI queries with realistic latencies and costs
- Realistic timestamps spread over the past 30 days

Usage:
    python scripts/generate_test_data.py --experiments 50 --runs 20 --queries 100
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add app directory to path for imports
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))

# Load environment variables from app/.env
from dotenv import load_dotenv
load_dotenv(app_dir / ".env")

from sqlalchemy import func

from src.services.db import (
    init_database,
    close_database,
    get_session,
    Experiment,
    OptimizationRun,
    AIQuery,
    ExperimentStatus,
    OptimizationMethod,
)


# Realistic parameter ranges for experiments
PARAMETER_RANGES = {
    "temperature": (200, 800),  # Celsius
    "pressure": (0.1, 10.0),  # atm
    "flow_rate": (10, 100),  # mL/min
    "concentration": (0.1, 5.0),  # M
    "ph": (1.0, 14.0),
    "reaction_time": (1, 120),  # minutes
}

# Realistic experimental contexts
CONTEXTS = [
    {"domain": "materials_synthesis", "target": "maximize_yield"},
    {"domain": "catalysis", "target": "minimize_byproducts"},
    {"domain": "organic_chemistry", "target": "optimize_selectivity"},
    {"domain": "polymer_science", "target": "maximize_molecular_weight"},
    {"domain": "nanoparticle_synthesis", "target": "control_size_distribution"},
]

# AI query templates
QUERY_TEMPLATES = [
    "How do I optimize {target} for {domain}?",
    "What parameters affect {target} most in {domain}?",
    "Suggest next experiment for {target} optimization",
    "Analyze results for {domain} experiment",
    "What is the relationship between temperature and {target}?",
    "How can I reduce noise in {domain} measurements?",
]

# User IDs for testing
USER_IDS = ["researcher_1", "researcher_2", "researcher_3", "lab_manager", "postdoc_5"]


def generate_experiment_id() -> str:
    """Generate realistic experiment ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    random_suffix = ''.join(random.choices('0123456789abcdef', k=6))
    return f"exp_{timestamp}_{random_suffix}"


def generate_run_id() -> str:
    """Generate realistic optimization run ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    random_suffix = ''.join(random.choices('0123456789abcdef', k=6))
    return f"run_{timestamp}_{random_suffix}"


def generate_query_id() -> str:
    """Generate realistic AI query ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    random_suffix = ''.join(random.choices('0123456789abcdef', k=6))
    return f"query_{timestamp}_{random_suffix}"


def generate_random_parameters(num_params: int = 3) -> dict:
    """Generate random experimental parameters."""
    param_names = random.sample(list(PARAMETER_RANGES.keys()), min(num_params, len(PARAMETER_RANGES)))
    parameters = {}
    for name in param_names:
        min_val, max_val = PARAMETER_RANGES[name]
        if isinstance(min_val, int):
            parameters[name] = random.randint(min_val, max_val)
        else:
            parameters[name] = round(random.uniform(min_val, max_val), 2)
    return parameters


def generate_random_timestamp(days_ago_max: int = 30) -> datetime:
    """Generate random timestamp within the past N days."""
    days_ago = random.uniform(0, days_ago_max)
    return datetime.utcnow() - timedelta(days=days_ago)


def generate_experiments(session, count: int, optimization_run_id: str = None) -> list:
    """Generate realistic experiments."""
    experiments = []
    
    for i in range(count):
        context = random.choice(CONTEXTS)
        parameters = generate_random_parameters()
        
        # Simulate realistic noise levels (higher at extreme temperatures)
        temp = parameters.get("temperature", 400)
        base_noise = 0.05
        temp_noise = abs(temp - 500) / 1000  # Higher noise at extremes
        noise_estimate = base_noise + temp_noise
        
        # Simulate measurement results
        # Assume yield is affected by parameters (simplified model)
        simulated_yield = 50 + random.gauss(0, 10)  # Base yield with noise
        simulated_yield = max(0, min(100, simulated_yield))  # Clamp to [0, 100]
        
        # Status distribution: 70% completed, 20% running, 10% failed
        status_roll = random.random()
        if status_roll < 0.7:
            status = ExperimentStatus.COMPLETED
            end_time = generate_random_timestamp(days_ago_max=30)
            start_time = end_time - timedelta(minutes=random.randint(30, 180))
            results = {
                "yield": round(simulated_yield, 2),
                "purity": round(random.uniform(85, 99), 2),
                "byproducts": round(random.uniform(0.1, 5.0), 2),
            }
            error_message = None
        elif status_roll < 0.9:
            status = ExperimentStatus.RUNNING
            start_time = generate_random_timestamp(days_ago_max=1)
            end_time = None
            results = None
            error_message = None
        else:
            status = ExperimentStatus.FAILED
            end_time = generate_random_timestamp(days_ago_max=30)
            start_time = end_time - timedelta(minutes=random.randint(10, 60))
            results = None
            error_message = random.choice([
                "Temperature sensor malfunction",
                "Pressure out of range",
                "Reagent contamination detected",
                "System timeout",
            ])
        
        experiment = Experiment(
            id=generate_experiment_id(),
            optimization_run_id=optimization_run_id,
            method=random.choice([m.value for m in OptimizationMethod]) if optimization_run_id else None,
            parameters=parameters,
            context=context,
            noise_estimate=round(noise_estimate, 4),
            results=results,
            status=status.value,
            start_time=start_time,
            end_time=end_time,
            error_message=error_message,
            created_by=random.choice(USER_IDS),
        )
        
        experiments.append(experiment)
    
    session.bulk_save_objects(experiments)
    session.commit()
    
    return experiments


def generate_optimization_runs(session, count: int, experiments_per_run: int = 10) -> list:
    """Generate realistic optimization runs with associated experiments."""
    runs = []
    
    for i in range(count):
        method = random.choice([m.value for m in OptimizationMethod])
        context = random.choice(CONTEXTS)
        
        # Status distribution: 60% completed, 30% running, 10% failed
        status_roll = random.random()
        if status_roll < 0.6:
            status = ExperimentStatus.COMPLETED
            end_time = generate_random_timestamp(days_ago_max=30)
            start_time = end_time - timedelta(hours=random.randint(4, 48))
            error_message = None
        elif status_roll < 0.9:
            status = ExperimentStatus.RUNNING
            start_time = generate_random_timestamp(days_ago_max=7)
            end_time = None
            error_message = None
        else:
            status = ExperimentStatus.FAILED
            end_time = generate_random_timestamp(days_ago_max=30)
            start_time = end_time - timedelta(hours=random.randint(1, 12))
            error_message = random.choice([
                "Optimizer convergence failure",
                "Resource limit exceeded",
                "Hardware connection lost",
            ])
        
        run = OptimizationRun(
            id=generate_run_id(),
            method=method,
            context=context,
            status=status.value,
            start_time=start_time,
            end_time=end_time,
            error_message=error_message,
            created_by=random.choice(USER_IDS),
        )
        
        runs.append(run)
        session.add(run)
        session.commit()
        
        # Generate experiments for this run
        if status != ExperimentStatus.FAILED:
            num_experiments = experiments_per_run if status == ExperimentStatus.COMPLETED else random.randint(1, experiments_per_run)
            generate_experiments(session, num_experiments, optimization_run_id=run.id)
    
    return runs


def generate_ai_queries(session, count: int) -> list:
    """Generate realistic AI queries."""
    queries = []
    
    for i in range(count):
        context = random.choice(CONTEXTS)
        query_template = random.choice(QUERY_TEMPLATES)
        query_text = query_template.format(**context)
        
        # Randomly select which model was chosen
        selected_model = random.choice(["flash", "pro", "adaptive_router"])
        
        # Simulate realistic latencies
        if selected_model == "flash":
            latency = random.gauss(800, 150)  # Fast model
        elif selected_model == "pro":
            latency = random.gauss(2500, 400)  # Slower but more accurate
        else:
            latency = random.gauss(3000, 500)  # Adaptive router has overhead
        
        latency = max(100, latency)  # Minimum 100ms
        
        # Simulate token usage (query + response)
        input_tokens = len(query_text.split()) * 1.3  # Approximate tokenization
        output_tokens = random.randint(100, 500)  # Response length
        
        # Calculate cost (approximate Gemini pricing)
        if selected_model == "flash":
            cost = (input_tokens * 0.00001 + output_tokens * 0.00003)  # Flash pricing
        else:
            cost = (input_tokens * 0.0001 + output_tokens * 0.0003)  # Pro pricing
        
        query = AIQuery(
            id=generate_query_id(),
            query=query_text,
            context=context,
            selected_model=selected_model,
            latency_ms=round(latency, 1),
            input_tokens=int(input_tokens),
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            created_by=random.choice(USER_IDS),
            created_at=generate_random_timestamp(days_ago_max=30),
        )
        
        queries.append(query)
    
    session.bulk_save_objects(queries)
    session.commit()
    
    return queries


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate test data for Cloud SQL database")
    parser.add_argument("--experiments", type=int, default=50, help="Number of standalone experiments to generate")
    parser.add_argument("--runs", type=int, default=20, help="Number of optimization runs to generate")
    parser.add_argument("--queries", type=int, default=100, help="Number of AI queries to generate")
    parser.add_argument("--experiments-per-run", type=int, default=10, help="Number of experiments per optimization run")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before generating new data")
    
    args = parser.parse_args()
    
    print("🔧 Initializing database connection...")
    init_database()
    
    session = get_session()
    
    try:
        if args.clear:
            print("🗑️  Clearing existing data...")
            session.query(Experiment).delete()
            session.query(OptimizationRun).delete()
            session.query(AIQuery).delete()
            session.commit()
            print("✅ Existing data cleared")
        
        print(f"\n📊 Generating test data...")
        print(f"  - {args.runs} optimization runs")
        print(f"  - {args.experiments_per_run} experiments per run")
        print(f"  - {args.experiments} standalone experiments")
        print(f"  - {args.queries} AI queries")
        
        # Generate optimization runs (with associated experiments)
        print("\n🔬 Generating optimization runs...")
        runs = generate_optimization_runs(session, args.runs, args.experiments_per_run)
        print(f"✅ Generated {len(runs)} optimization runs")
        
        # Generate standalone experiments
        print("\n🧪 Generating standalone experiments...")
        experiments = generate_experiments(session, args.experiments)
        print(f"✅ Generated {len(experiments)} standalone experiments")
        
        # Generate AI queries
        print("\n🤖 Generating AI queries...")
        queries = generate_ai_queries(session, args.queries)
        print(f"✅ Generated {len(queries)} AI queries")
        
        # Summary statistics
        total_experiments = session.query(Experiment).count()
        total_runs = session.query(OptimizationRun).count()
        total_queries = session.query(AIQuery).count()
        
        print(f"\n📈 Database Summary:")
        print(f"  Total Experiments: {total_experiments}")
        print(f"  Total Optimization Runs: {total_runs}")
        print(f"  Total AI Queries: {total_queries}")
        
        # Cost analysis
        total_cost = session.query(AIQuery).with_entities(
            func.sum(AIQuery.cost_usd)
        ).scalar() or 0
        
        print(f"\n💰 Cost Analysis:")
        print(f"  Total AI Cost: ${total_cost:.4f}")
        print(f"  Average Cost per Query: ${total_cost / total_queries:.6f}")
        
        print(f"\n✅ Test data generation complete!")
        print(f"\nYou can now test the metadata API endpoints:")
        print(f"  GET /api/experiments")
        print(f"  GET /api/optimization_runs")
        print(f"  GET /api/ai_queries")
    
    except Exception as e:
        print(f"\n❌ Error generating test data: {e}")
        session.rollback()
        raise
    
    finally:
        session.close()
        close_database()


if __name__ == "__main__":
    main()

