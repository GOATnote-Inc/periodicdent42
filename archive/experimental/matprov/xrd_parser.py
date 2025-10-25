"""
XRD Data Pipeline: Parse, normalize, hash, and store XRD patterns

Supports common formats:
- .xy (two-column ASCII)
- .xrdml (Bruker XML format)
- .csv (comma-separated values)

Features:
- Parse multiple formats
- Normalize to standard JSON
- Compute content hash (SHA-256)
- DVC integration ready
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class XRDPattern:
    """Standardized XRD pattern"""
    two_theta: List[float]  # 2θ angles (degrees)
    intensity: List[float]  # Intensity counts
    wavelength: float  # X-ray wavelength (Angstroms)
    step_size: Optional[float] = None  # Step size (degrees)
    scan_time: Optional[float] = None  # Total scan time (seconds)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'two_theta': self.two_theta,
            'intensity': self.intensity,
            'wavelength': self.wavelength,
            'step_size': self.step_size,
            'scan_time': self.scan_time,
            'metadata': self.metadata or {},
            'num_points': len(self.two_theta)
        }
    
    def compute_hash(self) -> str:
        """Compute SHA-256 content hash"""
        # Create deterministic representation
        data_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def save_json(self, output_path: Path) -> str:
        """Save to JSON file and return hash"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return self.compute_hash()


class XRDParser:
    """Parser for various XRD file formats"""
    
    @staticmethod
    def parse_xy(file_path: Path, wavelength: float = 1.5406) -> XRDPattern:
        """
        Parse .xy format (two-column ASCII)
        
        Format:
        2theta1 intensity1
        2theta2 intensity2
        ...
        
        Args:
            file_path: Path to .xy file
            wavelength: X-ray wavelength (default: Cu Kα = 1.5406 Å)
        """
        two_theta = []
        intensity = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        two_theta.append(float(parts[0]))
                        intensity.append(float(parts[1]))
                    except ValueError:
                        continue
        
        if not two_theta:
            raise ValueError(f"No valid data found in {file_path}")
        
        # Compute step size
        if len(two_theta) > 1:
            step_size = two_theta[1] - two_theta[0]
        else:
            step_size = None
        
        return XRDPattern(
            two_theta=two_theta,
            intensity=intensity,
            wavelength=wavelength,
            step_size=step_size,
            metadata={'source_file': str(file_path), 'format': 'xy'}
        )
    
    @staticmethod
    def parse_xrdml(file_path: Path) -> XRDPattern:
        """
        Parse .xrdml format (Bruker/PANalytical XML)
        
        Args:
            file_path: Path to .xrdml file
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Define namespace
        ns = {'xrd': 'http://www.xrdml.com/XRDMeasurement/1.5'}
        
        # Extract wavelength
        wavelength_elem = root.find('.//xrd:wavelength', ns)
        if wavelength_elem is not None:
            wavelength = float(wavelength_elem.text)
        else:
            wavelength = 1.5406  # Default Cu Kα
        
        # Extract 2θ and intensity
        positions = root.find('.//xrd:positions', ns)
        intensities = root.find('.//xrd:intensities', ns)
        
        if positions is None or intensities is None:
            raise ValueError(f"Invalid XRDML format in {file_path}")
        
        # Parse positions (2θ)
        start_pos = float(positions.find('xrd:startPosition', ns).text)
        end_pos = float(positions.find('xrd:endPosition', ns).text)
        
        # Parse intensities
        intensity_text = intensities.find('xrd:counts', ns).text
        intensity = [float(x) for x in intensity_text.split()]
        
        # Generate 2θ array
        num_points = len(intensity)
        two_theta = list(np.linspace(start_pos, end_pos, num_points))
        step_size = (end_pos - start_pos) / (num_points - 1) if num_points > 1 else None
        
        # Extract scan time
        scan_time_elem = root.find('.//xrd:timePerStep', ns)
        scan_time = float(scan_time_elem.text) * num_points if scan_time_elem is not None else None
        
        return XRDPattern(
            two_theta=two_theta,
            intensity=intensity,
            wavelength=wavelength,
            step_size=step_size,
            scan_time=scan_time,
            metadata={'source_file': str(file_path), 'format': 'xrdml'}
        )
    
    @staticmethod
    def parse_csv(file_path: Path, wavelength: float = 1.5406) -> XRDPattern:
        """
        Parse CSV format
        
        Expected columns: two_theta, intensity
        Or: 2theta, intensity
        
        Args:
            file_path: Path to CSV file
            wavelength: X-ray wavelength (default: Cu Kα)
        """
        import csv
        
        two_theta = []
        intensity = []
        
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            
            # Detect column names
            if reader.fieldnames is None:
                raise ValueError(f"No header found in {file_path}")
            
            # Try to find 2θ and intensity columns
            theta_col = None
            intensity_col = None
            
            for col in reader.fieldnames:
                col_lower = col.lower().strip()
                if 'theta' in col_lower or '2θ' in col_lower or '2theta' in col_lower:
                    theta_col = col
                elif 'intensity' in col_lower or 'counts' in col_lower:
                    intensity_col = col
            
            if not theta_col or not intensity_col:
                raise ValueError(f"Could not find 2θ and intensity columns in {file_path}")
            
            for row in reader:
                try:
                    two_theta.append(float(row[theta_col]))
                    intensity.append(float(row[intensity_col]))
                except (ValueError, KeyError):
                    continue
        
        if not two_theta:
            raise ValueError(f"No valid data found in {file_path}")
        
        step_size = two_theta[1] - two_theta[0] if len(two_theta) > 1 else None
        
        return XRDPattern(
            two_theta=two_theta,
            intensity=intensity,
            wavelength=wavelength,
            step_size=step_size,
            metadata={'source_file': str(file_path), 'format': 'csv'}
        )
    
    @staticmethod
    def parse(file_path: Path, wavelength: Optional[float] = None) -> XRDPattern:
        """
        Auto-detect format and parse
        
        Args:
            file_path: Path to XRD file
            wavelength: Optional wavelength override
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.xy':
            pattern = XRDParser.parse_xy(file_path, wavelength or 1.5406)
        elif suffix == '.xrdml':
            pattern = XRDParser.parse_xrdml(file_path)
            if wavelength:
                pattern.wavelength = wavelength
        elif suffix == '.csv':
            pattern = XRDParser.parse_csv(file_path, wavelength or 1.5406)
        else:
            # Try XY format as fallback
            pattern = XRDParser.parse_xy(file_path, wavelength or 1.5406)
        
        return pattern


def normalize_xrd(pattern: XRDPattern) -> XRDPattern:
    """
    Normalize XRD pattern
    
    - Scale intensity to [0, 1]
    - Interpolate to standard 2θ grid (10-90°, 0.02° steps)
    """
    from scipy.interpolate import interp1d
    
    # Create standard grid
    standard_two_theta = np.arange(10.0, 90.0, 0.02)
    
    # Interpolate to standard grid
    if len(pattern.two_theta) > 1:
        f = interp1d(
            pattern.two_theta,
            pattern.intensity,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        standard_intensity = f(standard_two_theta)
    else:
        standard_intensity = np.zeros_like(standard_two_theta)
    
    # Normalize to [0, 1]
    max_intensity = np.max(standard_intensity)
    if max_intensity > 0:
        standard_intensity = standard_intensity / max_intensity
    
    return XRDPattern(
        two_theta=standard_two_theta.tolist(),
        intensity=standard_intensity.tolist(),
        wavelength=pattern.wavelength,
        step_size=0.02,
        metadata={**pattern.metadata, 'normalized': True}
    )


# Example usage
if __name__ == "__main__":
    import tempfile
    
    # Create sample XY file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xy', delete=False) as f:
        f.write("# Sample XRD pattern\n")
        f.write("10.0 100\n")
        f.write("20.0 500\n")
        f.write("30.0 1000\n")
        f.write("40.0 300\n")
        f.write("50.0 150\n")
        temp_path = Path(f.name)
    
    print("=== XRD Parser Demo ===\n")
    
    # Parse
    pattern = XRDParser.parse(temp_path)
    print(f"✅ Parsed {temp_path}")
    print(f"   Points: {len(pattern.two_theta)}")
    print(f"   2θ range: {min(pattern.two_theta):.2f}° - {max(pattern.two_theta):.2f}°")
    print(f"   Max intensity: {max(pattern.intensity):.0f}")
    print(f"   Wavelength: {pattern.wavelength} Å")
    
    # Compute hash
    hash_val = pattern.compute_hash()
    print(f"\n✅ Content hash: {hash_val[:16]}...")
    
    # Save to JSON
    output_path = temp_path.with_suffix('.json')
    pattern.save_json(output_path)
    print(f"✅ Saved to: {output_path}")
    
    # Cleanup
    temp_path.unlink()
    output_path.unlink()
    
    print("\n✅ XRD pipeline complete!")

