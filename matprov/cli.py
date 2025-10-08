"""
matprov CLI: Materials Provenance Tracking Command-Line Interface

Wraps DVC for multi-GB file tracking, implements Merkle trees for experiment
lineage, and integrates with Sigstore for keyless signing.
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from matprov.schema import MaterialsExperiment
from matprov.provenance import ProvenanceTracker


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    matprov: Materials Provenance Tracking System
    
    Track materials synthesis experiments with cryptographic provenance.
    """
    pass


@cli.command()
@click.option('--dvc-remote', default=None, help='DVC remote storage URL (optional)')
@click.option('--git-init', is_flag=True, help='Initialize git repository if not present')
def init(dvc_remote: Optional[str], git_init: bool):
    """Initialize tracking in the current directory."""
    cwd = Path.cwd()
    click.echo(f"🔧 Initializing matprov in: {cwd}")
    
    # Check for git
    git_dir = cwd / ".git"
    if not git_dir.exists():
        if git_init:
            import subprocess
            subprocess.run(["git", "init"], check=True)
            click.echo("✅ Initialized git repository")
        else:
            click.echo("⚠️  No git repository found. Run with --git-init or init git manually.", err=True)
            sys.exit(1)
    
    # Initialize DVC
    dvc_dir = cwd / ".dvc"
    if not dvc_dir.exists():
        import subprocess
        subprocess.run(["dvc", "init"], check=True)
        click.echo("✅ Initialized DVC")
    else:
        click.echo("⚠️  DVC already initialized")
    
    # Add remote if specified
    if dvc_remote:
        import subprocess
        subprocess.run(["dvc", "remote", "add", "-d", "storage", dvc_remote], check=False)
        click.echo(f"✅ Added DVC remote: {dvc_remote}")
    
    # Create matprov directories
    matprov_dir = cwd / ".matprov"
    matprov_dir.mkdir(exist_ok=True)
    
    experiments_dir = cwd / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    
    data_dir = cwd / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Initialize provenance tracker
    tracker = ProvenanceTracker(matprov_dir)
    click.echo(f"✅ Created provenance ledger: {matprov_dir / 'ledger.jsonl'}")
    
    # Create README
    readme = experiments_dir / "README.md"
    if not readme.exists():
        readme.write_text("""# Experiments

This directory contains materials synthesis experiment records tracked by matprov.

Each experiment is stored as a JSON file following the MaterialsExperiment schema.

Use `matprov track-experiment <file>` to add experiments to the provenance ledger.
""")
        click.echo(f"✅ Created: {readme}")
    
    click.echo("\n✅ matprov initialization complete!")
    click.echo("\nNext steps:")
    click.echo("  1. Create an experiment JSON file")
    click.echo("  2. Track it: matprov track-experiment <file>")
    click.echo("  3. Verify: matprov verify <exp_id>")


@cli.command()
@click.argument('exp_json', type=click.Path(exists=True, path_type=Path))
@click.option('--dvc-add', is_flag=True, help='Automatically add large files to DVC')
def track_experiment(exp_json: Path, dvc_add: bool):
    """Add experiment with auto-hashing to provenance ledger."""
    click.echo(f"📝 Tracking experiment: {exp_json}")
    
    # Load experiment
    try:
        experiment = MaterialsExperiment.from_json(exp_json)
        click.echo(f"✅ Loaded experiment: {experiment.metadata.experiment_id}")
    except Exception as e:
        click.echo(f"❌ Failed to load experiment: {e}", err=True)
        sys.exit(1)
    
    # Compute content hash
    content_hash = experiment.content_hash()
    click.echo(f"🔐 Content hash: {content_hash}")
    
    # Track large files with DVC if requested
    if dvc_add:
        files_to_track = []
        
        # Collect file paths from characterization data
        if experiment.characterization.xrd:
            files_to_track.append(experiment.characterization.xrd.file_path)
        if experiment.characterization.cif_file_path:
            files_to_track.append(experiment.characterization.cif_file_path)
        files_to_track.extend(experiment.characterization.sem_images)
        files_to_track.extend(experiment.characterization.eds_spectra)
        
        for file_path in files_to_track:
            if Path(file_path).exists():
                import subprocess
                result = subprocess.run(
                    ["dvc", "add", file_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    click.echo(f"  ✅ Added to DVC: {file_path}")
                else:
                    click.echo(f"  ⚠️  Failed to add {file_path}: {result.stderr.strip()}")
    
    # Add to provenance ledger
    matprov_dir = Path.cwd() / ".matprov"
    if not matprov_dir.exists():
        click.echo("❌ matprov not initialized. Run 'matprov init' first.", err=True)
        sys.exit(1)
    
    tracker = ProvenanceTracker(matprov_dir)
    entry = tracker.add_experiment(
        experiment_id=experiment.metadata.experiment_id,
        content_hash=content_hash,
        file_path=str(exp_json),
        metadata={
            "operator": experiment.metadata.operator,
            "target_formula": experiment.synthesis.target_formula,
            "outcome_status": experiment.outcome.status.value,
        }
    )
    
    click.echo(f"✅ Added to provenance ledger")
    click.echo(f"   Merkle root: {entry['merkle_root']}")
    click.echo(f"   Timestamp: {entry['timestamp']}")


@cli.command()
@click.argument('pred_id')
@click.argument('exp_id')
@click.option('--metadata', default='{}', help='Additional metadata as JSON string')
def link_prediction(pred_id: str, exp_id: str, metadata: str):
    """Create prediction → experiment link."""
    click.echo(f"🔗 Linking prediction {pred_id} → experiment {exp_id}")
    
    matprov_dir = Path.cwd() / ".matprov"
    if not matprov_dir.exists():
        click.echo("❌ matprov not initialized. Run 'matprov init' first.", err=True)
        sys.exit(1)
    
    tracker = ProvenanceTracker(matprov_dir)
    
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError as e:
        click.echo(f"❌ Invalid metadata JSON: {e}", err=True)
        sys.exit(1)
    
    entry = tracker.add_link(
        link_type="prediction→experiment",
        source_id=pred_id,
        target_id=exp_id,
        metadata=metadata_dict
    )
    
    click.echo(f"✅ Link created")
    click.echo(f"   Merkle root: {entry['merkle_root']}")


@cli.command()
@click.argument('exp_id')
def verify(exp_id: str):
    """Check cryptographic signatures for an experiment."""
    click.echo(f"🔍 Verifying experiment: {exp_id}")
    
    matprov_dir = Path.cwd() / ".matprov"
    if not matprov_dir.exists():
        click.echo("❌ matprov not initialized.", err=True)
        sys.exit(1)
    
    tracker = ProvenanceTracker(matprov_dir)
    
    # Find experiment in ledger
    entry = tracker.find_experiment(exp_id)
    if not entry:
        click.echo(f"❌ Experiment not found in ledger", err=True)
        sys.exit(1)
    
    click.echo(f"✅ Found in ledger:")
    click.echo(f"   Content hash: {entry['content_hash']}")
    click.echo(f"   Merkle root: {entry['merkle_root']}")
    click.echo(f"   Timestamp: {entry['timestamp']}")
    
    # Verify Merkle chain
    is_valid = tracker.verify_ledger()
    if is_valid:
        click.echo(f"✅ Merkle chain verified")
    else:
        click.echo(f"❌ Merkle chain verification failed!", err=True)
        sys.exit(1)
    
    # Check file integrity if path exists
    if 'file_path' in entry:
        file_path = Path(entry['file_path'])
        if file_path.exists():
            exp = MaterialsExperiment.from_json(file_path)
            current_hash = exp.content_hash()
            if current_hash == entry['content_hash']:
                click.echo(f"✅ File integrity verified")
            else:
                click.echo(f"⚠️  File has been modified!")
                click.echo(f"   Expected: {entry['content_hash']}")
                click.echo(f"   Current:  {current_hash}")
        else:
            click.echo(f"⚠️  File not found: {file_path}")


@cli.command()
@click.argument('exp_id')
@click.option('--format', 'output_format', type=click.Choice(['tree', 'json', 'dot']), default='tree')
@click.option('--output', type=click.Path(path_type=Path), help='Output file (optional)')
def lineage(exp_id: str, output_format: str, output: Optional[Path]):
    """Show full experiment history tree."""
    click.echo(f"🌳 Tracing lineage for: {exp_id}")
    
    matprov_dir = Path.cwd() / ".matprov"
    if not matprov_dir.exists():
        click.echo("❌ matprov not initialized.", err=True)
        sys.exit(1)
    
    tracker = ProvenanceTracker(matprov_dir)
    lineage_data = tracker.get_lineage(exp_id)
    
    if not lineage_data:
        click.echo(f"❌ No lineage found for {exp_id}", err=True)
        sys.exit(1)
    
    if output_format == 'json':
        output_text = json.dumps(lineage_data, indent=2)
    elif output_format == 'dot':
        output_text = tracker.lineage_to_dot(lineage_data)
    else:  # tree
        output_text = tracker.lineage_to_tree(lineage_data)
    
    if output:
        output.write_text(output_text)
        click.echo(f"✅ Lineage saved to: {output}")
    else:
        click.echo(output_text)


@cli.command()
def status():
    """Show matprov repository status."""
    matprov_dir = Path.cwd() / ".matprov"
    
    if not matprov_dir.exists():
        click.echo("❌ matprov not initialized in this directory")
        click.echo("   Run 'matprov init' to get started")
        sys.exit(1)
    
    tracker = ProvenanceTracker(matprov_dir)
    stats = tracker.get_statistics()
    
    click.echo("📊 matprov Status")
    click.echo(f"   Experiments: {stats['total_experiments']}")
    click.echo(f"   Links: {stats['total_links']}")
    click.echo(f"   Ledger entries: {stats['total_entries']}")
    click.echo(f"   Current Merkle root: {stats['current_merkle_root'][:16]}...")
    click.echo(f"   Ledger verified: {'✅' if stats['ledger_valid'] else '❌'}")


if __name__ == "__main__":
    cli()

