"""
Materials Project API Connector

Query, download, and store crystal structures with full provenance tracking.

Features:
- Query MP database by formula, properties
- Download CIF structures
- Store with DVC tracking
- Link to matprov experiments
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import json
import os


class MaterialsProjectConnector:
    """Connector for Materials Project API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Materials Project connector
        
        Args:
            api_key: MP API key (or set MP_API_KEY environment variable)
        """
        self.api_key = api_key or os.getenv('MP_API_KEY')
        
        if not self.api_key:
            print("⚠️  No MP API key provided.")
            print("   Get one at: https://next-gen.materialsproject.org/api")
            print("   Set via: export MP_API_KEY=your_key")
            self._mp_available = False
        else:
            try:
                from mp_api.client import MPRester
                self.mpr = MPRester(self.api_key)
                self._mp_available = True
                print("✅ Connected to Materials Project")
            except ImportError:
                print("⚠️  mp-api not installed. Install with: pip install mp-api")
                self._mp_available = False
            except Exception as e:
                print(f"⚠️  Error connecting to MP: {e}")
                self._mp_available = False
    
    def search_by_formula(
        self,
        formula: str,
        is_stable: bool = True,
        properties: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Materials Project by chemical formula
        
        Args:
            formula: Chemical formula (e.g., "YBa2Cu3O7")
            is_stable: Only return stable materials
            properties: List of properties to retrieve
        
        Returns:
            List of material dictionaries
        """
        if not self._mp_available:
            return []
        
        try:
            # Default properties
            if properties is None:
                properties = [
                    'material_id',
                    'formula_pretty',
                    'structure',
                    'energy_per_atom',
                    'formation_energy_per_atom',
                    'is_stable',
                    'band_gap',
                    'density'
                ]
            
            # Query
            docs = self.mpr.materials.summary.search(
                formula=formula,
                is_stable=is_stable,
                fields=properties
            )
            
            results = []
            for doc in docs:
                result = {
                    'material_id': str(doc.material_id),
                    'formula': doc.formula_pretty,
                    'is_stable': doc.is_stable,
                    'energy_per_atom': doc.energy_per_atom,
                    'band_gap': doc.band_gap,
                    'density': doc.density
                }
                
                if hasattr(doc, 'structure') and doc.structure:
                    result['structure'] = doc.structure
                
                results.append(result)
            
            return results
        
        except Exception as e:
            print(f"❌ Error searching MP: {e}")
            return []
    
    def get_structure(self, material_id: str) -> Optional[Any]:
        """
        Get crystal structure by MP ID
        
        Args:
            material_id: Materials Project ID (e.g., "mp-12345")
        
        Returns:
            pymatgen Structure object
        """
        if not self._mp_available:
            return None
        
        try:
            doc = self.mpr.materials.summary.get_data_by_id(material_id)
            return doc.structure if doc else None
        except Exception as e:
            print(f"❌ Error fetching structure: {e}")
            return None
    
    def download_cif(
        self,
        material_id: str,
        output_dir: Path = Path("data/cif"),
        add_to_dvc: bool = False
    ) -> Optional[Path]:
        """
        Download CIF file for a material
        
        Args:
            material_id: Materials Project ID
            output_dir: Directory to save CIF
            add_to_dvc: Whether to add to DVC tracking
        
        Returns:
            Path to downloaded CIF file
        """
        if not self._mp_available:
            return None
        
        try:
            structure = self.get_structure(material_id)
            if not structure:
                return None
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CIF
            from pymatgen.io.cif import CifWriter
            cif_path = output_dir / f"{material_id}.cif"
            writer = CifWriter(structure)
            writer.write_file(str(cif_path))
            
            # Compute hash
            cif_hash = self._compute_file_hash(cif_path)
            
            print(f"✅ Downloaded: {cif_path}")
            print(f"   Formula: {structure.composition.reduced_formula}")
            print(f"   Hash: {cif_hash[:16]}...")
            
            # Add to DVC if requested
            if add_to_dvc:
                import subprocess
                try:
                    subprocess.run(['dvc', 'add', str(cif_path)], check=True, capture_output=True)
                    print(f"✅ Added to DVC: {cif_path}.dvc")
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"⚠️  DVC error: {e}")
            
            # Save metadata
            metadata = {
                'material_id': material_id,
                'formula': structure.composition.reduced_formula,
                'cif_path': str(cif_path),
                'cif_hash': cif_hash,
                'source': 'Materials Project',
                'num_sites': len(structure.sites),
                'density': structure.density,
                'volume': structure.lattice.volume
            }
            
            metadata_path = cif_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return cif_path
        
        except Exception as e:
            print(f"❌ Error downloading CIF: {e}")
            return None
    
    def batch_download(
        self,
        material_ids: List[str],
        output_dir: Path = Path("data/cif"),
        add_to_dvc: bool = False
    ) -> List[Path]:
        """
        Download multiple CIF files
        
        Args:
            material_ids: List of MP IDs
            output_dir: Directory to save CIFs
            add_to_dvc: Whether to add to DVC
        
        Returns:
            List of downloaded file paths
        """
        downloaded = []
        
        for mp_id in material_ids:
            print(f"\n📥 Downloading {mp_id}...")
            cif_path = self.download_cif(mp_id, output_dir, add_to_dvc)
            if cif_path:
                downloaded.append(cif_path)
        
        print(f"\n✅ Downloaded {len(downloaded)}/{len(material_ids)} files")
        return downloaded
    
    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of file"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()


# Example usage
if __name__ == "__main__":
    import tempfile
    
    print("=== Materials Project Connector Demo ===\n")
    
    # Initialize connector
    connector = MaterialsProjectConnector()
    
    if not connector._mp_available:
        print("\n⚠️  Demo requires MP API key")
        print("   1. Get API key: https://next-gen.materialsproject.org/api")
        print("   2. Set: export MP_API_KEY=your_key")
        print("   3. Rerun: python matprov/materials_project.py")
        
        # Show mock example
        print("\n📝 Mock Example (no API call):")
        print("   Formula: YBa2Cu3O7")
        print("   Results: [")
        print("     {")
        print("       'material_id': 'mp-12345',")
        print("       'formula': 'YBa2Cu3O7',")
        print("       'is_stable': True,")
        print("       'band_gap': 0.0,")
        print("       'density': 6.38")
        print("     }")
        print("   ]")
        print("\n   Download: mp-12345.cif")
        print("   Hash: abc123...")
        print("   DVC: mp-12345.cif.dvc")
    else:
        # Search for YBCO superconductor
        formula = "YBa2Cu3O7"
        print(f"🔍 Searching for: {formula}")
        
        results = connector.search_by_formula(formula, is_stable=True)
        
        if results:
            print(f"\n✅ Found {len(results)} materials:")
            for result in results[:3]:  # Show first 3
                print(f"   - {result['material_id']}: {result['formula']}")
                print(f"     Band Gap: {result['band_gap']} eV")
                print(f"     Density: {result['density']:.2f} g/cm³")
            
            # Download first result
            if results:
                mp_id = results[0]['material_id']
                print(f"\n📥 Downloading structure for {mp_id}...")
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    cif_path = connector.download_cif(
                        mp_id,
                        output_dir=Path(tmpdir),
                        add_to_dvc=False
                    )
                    
                    if cif_path:
                        print(f"✅ Saved to: {cif_path}")
                        print(f"   File size: {cif_path.stat().st_size} bytes")
        else:
            print(f"❌ No results found for {formula}")
    
    print("\n✅ Materials Project connector demo complete!")

