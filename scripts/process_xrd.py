#!/usr/bin/env python3
"""
XRD Processing CLI

Parse, normalize, and store XRD patterns with DVC tracking.

Usage:
    python scripts/process_xrd.py data/xrd/sample.xy --normalize --dvc
"""

import click
import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

from matprov.xrd_parser import XRDParser, normalize_xrd


@click.command()
@click.argument('xrd_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output JSON path (default: same name with .json)')
@click.option('--normalize', is_flag=True, help='Normalize to standard grid')
@click.option('--wavelength', type=float, help='X-ray wavelength in Angstroms (default: 1.5406 for Cu Kα)')
@click.option('--dvc', is_flag=True, help='Add to DVC tracking')
@click.option('--dvc-remote', help='DVC remote to push to')
def process_xrd(xrd_file, output, normalize, wavelength, dvc, dvc_remote):
    """Process XRD file and store with provenance"""
    
    xrd_path = Path(xrd_file)
    
    click.echo(f"📊 Processing XRD: {xrd_path}")
    
    # Parse
    try:
        pattern = XRDParser.parse(xrd_path, wavelength=wavelength)
        click.echo(f"✅ Parsed successfully")
        click.echo(f"   Format: {pattern.metadata.get('format', 'unknown')}")
        click.echo(f"   Points: {len(pattern.two_theta)}")
        click.echo(f"   2θ range: {min(pattern.two_theta):.2f}° - {max(pattern.two_theta):.2f}°")
        click.echo(f"   Wavelength: {pattern.wavelength} Å")
    except Exception as e:
        click.echo(f"❌ Error parsing: {e}", err=True)
        sys.exit(1)
    
    # Normalize if requested
    if normalize:
        click.echo(f"\n🔧 Normalizing...")
        try:
            pattern = normalize_xrd(pattern)
            click.echo(f"✅ Normalized to standard grid")
            click.echo(f"   New points: {len(pattern.two_theta)}")
            click.echo(f"   Step size: {pattern.step_size}°")
        except Exception as e:
            click.echo(f"❌ Error normalizing: {e}", err=True)
            sys.exit(1)
    
    # Determine output path
    if not output:
        output = xrd_path.with_suffix('.json')
    else:
        output = Path(output)
    
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    try:
        content_hash = pattern.save_json(output)
        click.echo(f"\n✅ Saved to: {output}")
        click.echo(f"🔐 Content hash: {content_hash[:16]}...")
    except Exception as e:
        click.echo(f"❌ Error saving: {e}", err=True)
        sys.exit(1)
    
    # DVC tracking
    if dvc:
        click.echo(f"\n📦 Adding to DVC...")
        try:
            # Add to DVC
            subprocess.run(['dvc', 'add', str(output)], check=True, capture_output=True)
            click.echo(f"✅ Added to DVC: {output}.dvc")
            
            # Push to remote if specified
            if dvc_remote:
                subprocess.run(['dvc', 'push', str(output), '-r', dvc_remote], check=True, capture_output=True)
                click.echo(f"✅ Pushed to remote: {dvc_remote}")
            
            # Git add the .dvc file
            subprocess.run(['git', 'add', f"{output}.dvc"], check=True, capture_output=True)
            click.echo(f"✅ Staged DVC file for git")
            
        except subprocess.CalledProcessError as e:
            click.echo(f"⚠️  DVC error: {e.stderr.decode() if e.stderr else str(e)}", err=True)
        except FileNotFoundError:
            click.echo(f"⚠️  DVC not installed. Install with: pip install dvc", err=True)
    
    click.echo(f"\n✅ XRD processing complete!")
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Link to experiment: matprov track-experiment --xrd {output}")
    click.echo(f"  2. Verify hash: sha256sum {output}")
    click.echo(f"  3. Commit: git commit -m 'Add XRD pattern {content_hash[:8]}'")


if __name__ == '__main__':
    process_xrd()

