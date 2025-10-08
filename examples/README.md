# Example CIF Files for BETE-NET Testing

This directory contains example crystal structures for testing BETE-NET superconductor screening.

## Files

### Nb.cif - Niobium (mp-48)
- **Formula**: Nb
- **Structure**: Body-centered cubic (BCC)
- **Lattice Parameter**: a = 3.301 Å
- **Space Group**: Im-3m (229)
- **Experimental Tc**: 9.2 K
- **Expected λ**: ~1.0
- **Expected ⟨ω_log⟩**: ~250 K

**Use Case**: Golden test for typical BCS superconductor

## Testing Examples

### CLI Testing

```bash
# Single prediction
bete-screen infer --cif examples/Nb.cif --mu-star 0.10

# With evidence pack
bete-screen infer --cif examples/Nb.cif --mu-star 0.10 --evidence

# Batch screening
echo "cif_path" > test_batch.csv
echo "examples/Nb.cif" >> test_batch.csv
bete-screen screen --csv test_batch.csv --out results.parquet
```

### API Testing

```bash
# Start server
cd app && ./start_server.sh

# Test prediction
curl -X POST http://localhost:8080/api/bete/predict \
  -H "Content-Type: application/json" \
  -d "{\"cif_content\": \"$(cat examples/Nb.cif)\", \"mu_star\": 0.10}" | jq
```

### Python Testing

```python
from app.src.bete_net_io.inference import predict_tc
from pathlib import Path

# Predict Tc
prediction = predict_tc("examples/Nb.cif", mu_star=0.10)

print(f"Formula: {prediction.formula}")
print(f"Tc: {prediction.tc_kelvin:.2f} ± {prediction.tc_std:.2f} K")
print(f"λ: {prediction.lambda_ep:.3f} ± {prediction.lambda_std:.3f}")
```

## Adding More Examples

To add more test structures:

1. **Download from Materials Project**:
   ```python
   from pymatgen.ext.matproj import MPRester
   
   with MPRester() as mpr:
       structure = mpr.get_structure_by_material_id("mp-66")  # MgB2
       structure.to(filename="examples/MgB2.cif")
   ```

2. **Verify structure**:
   ```bash
   bete-screen infer --cif examples/MgB2.cif --mu-star 0.10
   ```

3. **Update this README** with expected values

## Golden Test Materials

For comprehensive validation, add these materials:

| Material | MP-ID | Formula | Tc_exp (K) | Structure |
|----------|-------|---------|------------|-----------|
| Niobium | mp-48 | Nb | 9.2 | BCC |
| Magnesium Diboride | mp-763 | MgB2 | 39 | Hexagonal |
| Aluminum | mp-134 | Al | 1.2 | FCC |
| Lead | mp-20483 | Pb | 7.2 | FCC |
| Tin (β) | mp-117 | Sn | 3.7 | Tetragonal |

---

© 2025 GOATnote Autonomous Research Lab Initiative  
Licensed under Apache 2.0

