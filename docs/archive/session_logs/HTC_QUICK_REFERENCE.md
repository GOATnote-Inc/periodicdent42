# HTC Quick Reference Guide

**Last Updated**: October 10, 2025  
**Status**: Production-deployed (revision ard-backend-00053-xcp)

---

## Quick Commands

### Test Production API

```bash
# Health check
curl https://ard-backend-dydzexswua-uc.a.run.app/api/htc/health | jq

# Predict Tc for MgB2
curl -X POST https://ard-backend-dydzexswua-uc.a.run.app/api/htc/predict \
  -H "Content-Type: application/json" \
  -d '{"composition": "MgB2", "pressure_gpa": 0.0}' | jq

# Run 21-material test suite
./scripts/test_htc_materials.sh
```

### Deploy New Version

```bash
# 1. Build Docker image
cd app
gcloud builds submit --tag gcr.io/periodicdent42/ard-backend:latest .

# 2. Deploy to Cloud Run
gcloud run deploy ard-backend \
  --image gcr.io/periodicdent42/ard-backend:latest \
  --region us-central1 \
  --quiet

# 3. Check deployment status
gcloud run services describe ard-backend --region us-central1 --format json | jq '.status.conditions'
```

### Database Operations

```bash
# Check Cloud SQL Proxy status
ps aux | grep cloud-sql-proxy

# Connect to database
export PGPASSWORD=ard_secure_password_2024
psql -h localhost -p 5433 -U ard_user -d ard_intelligence

# Query HTC predictions
SELECT composition, tc_predicted, created_at 
FROM htc_predictions 
ORDER BY created_at DESC 
LIMIT 10;

# Apply migrations
cd app
alembic upgrade head
```

### Run Tests Locally

```bash
# Unit tests only
PYTHONPATH=/Users/kiteboard/periodicdent42 pytest app/tests/test_htc_domain.py -v

# All HTC tests
PYTHONPATH=/Users/kiteboard/periodicdent42 pytest app/tests/test_htc_*.py -v

# With coverage
PYTHONPATH=/Users/kiteboard/periodicdent42 pytest app/tests/ --cov=app/src/htc --cov-report=term-missing
```

---

## Key Files

### Core Modules

```
app/src/htc/
├── __init__.py                  # Package exports
├── domain.py                    # SuperconductorPredictor + Allen-Dynes
├── structure_utils.py           # Composition → Structure + property estimation
├── runner.py                    # Experiment orchestration
├── uncertainty.py               # ISO GUM + Sobol (stub)
├── visualization.py             # Plots (stub)
└── validation.py                # Test suite (stub)
```

### API

```
app/src/api/
├── main.py                      # FastAPI app (includes htc_router)
└── htc_api.py                   # HTC endpoints (/predict, /screen, etc.)
```

### Database

```
app/src/services/
└── db.py                        # HTCPrediction ORM model

app/alembic/versions/
└── 003_add_htc_predictions.py   # Migration for htc_predictions table
```

### Tests

```
app/tests/
├── test_htc_domain.py           # Unit tests (15+ tests)
├── test_htc_api.py              # Integration tests
└── test_htc_integration.py      # End-to-end tests

scripts/
└── test_htc_materials.sh        # 21-material benchmark suite
```

### Documentation

```
docs/
├── HTC_INTEGRATION.md           # Integration guide (500 lines)
├── HTC_API_EXAMPLES.md          # API usage examples (300 lines)
└── PRODUCTION_MONITORING.md     # Operations guide (400 lines)

HTC_PHYSICS_CALIBRATION_STATUS_OCT10_2025.md    # Physics status (450 lines)
HTC_PRODUCTION_MILESTONE_OCT10_2025.md          # Executive summary (900 lines)
```

---

## API Endpoints

### Production URL
`https://ard-backend-dydzexswua-uc.a.run.app`

### Endpoints

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/api/htc/health` | GET | Health check + dependencies | None |
| `/api/htc/predict` | POST | Tc prediction with uncertainty | None |
| `/api/htc/screen` | POST | High-throughput screening | None |
| `/api/htc/optimize` | POST | Multi-objective optimization | None |
| `/api/htc/validate` | POST | Benchmark validation | None |
| `/api/htc/results/{run_id}` | GET | Results retrieval | None |

### Example Request

```bash
curl -X POST https://ard-backend-dydzexswua-uc.a.run.app/api/htc/predict \
  -H "Content-Type: application/json" \
  -d '{
    "composition": "Nb3Sn",
    "pressure_gpa": 0.0,
    "include_uncertainty": true
  }' | jq
```

### Example Response

```json
{
  "composition": "Nb3Sn",
  "reduced_formula": "Nb3Sn",
  "tc_predicted": 4.33,
  "tc_lower_95ci": 3.49,
  "tc_upper_95ci": 5.17,
  "tc_uncertainty": 0.43,
  "pressure_required_gpa": 0.0,
  "lambda_ep": 0.68,
  "omega_log": 254.0,
  "xi_parameter": 0.40,
  "phonon_stable": true,
  "thermo_stable": true,
  "confidence_level": "medium",
  "timestamp": "2025-10-10T15:45:00"
}
```

---

## Database Schema

### `htc_predictions` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | String (UUID) | Primary key |
| `composition` | String | Full composition (e.g., "Nb3Sn") |
| `reduced_formula` | String | Reduced formula |
| `structure_info` | JSON | Pymatgen structure data |
| `tc_predicted` | Float | Predicted Tc (K) |
| `tc_lower_95ci` | Float | Lower 95% confidence interval |
| `tc_upper_95ci` | Float | Upper 95% confidence interval |
| `tc_uncertainty` | Float | Standard deviation (K) |
| `pressure_required_gpa` | Float | Pressure (GPa) |
| `lambda_ep` | Float | Electron-phonon coupling |
| `omega_log` | Float | Phonon frequency (K) |
| `xi_parameter` | Float | Stability indicator |
| `phonon_stable` | String | "true"/"false" |
| `thermo_stable` | String | "true"/"false" |
| `prediction_method` | String | "McMillan-Allen-Dynes" |
| `confidence_level` | String | "low"/"medium"/"high" |
| `experiment_id` | String | Link to experiment run |
| `created_by` | String | User identifier |
| `created_at` | DateTime | Timestamp |

### Query Examples

```sql
-- Get all predictions for MgB2
SELECT * FROM htc_predictions WHERE composition LIKE '%MgB2%';

-- Get recent high-Tc predictions
SELECT composition, tc_predicted, created_at 
FROM htc_predictions 
WHERE tc_predicted > 20.0 
ORDER BY created_at DESC;

-- Count predictions by confidence level
SELECT confidence_level, COUNT(*) 
FROM htc_predictions 
GROUP BY confidence_level;
```

---

## Physics Formulas

### Allen-Dynes Tc Formula

```python
def allen_dynes_tc(lambda_ep: float, omega_log: float, mu_star: float = 0.13) -> float:
    """
    McMillan-Allen-Dynes formula for BCS superconductors.
    
    Parameters:
        lambda_ep: Electron-phonon coupling constant
        omega_log: Logarithmic phonon frequency (K)
        mu_star: Coulomb pseudopotential (default: 0.13)
    
    Returns:
        tc: Critical temperature (K)
    """
    if lambda_ep <= mu_star:
        return 0.0
    
    f1 = 1.0 + (lambda_ep / 2.46) * (1.0 + 3.8 * mu_star)
    f2 = 1.0 + (lambda_ep**1.5 / (lambda_ep + 0.0002))
    
    exponent = -1.04 * (1.0 + lambda_ep) / (lambda_ep - mu_star * (1.0 + 0.62 * lambda_ep))
    
    tc = (omega_log / 1.2) * np.exp(exponent) * f1 / f2
    return float(tc)
```

### Empirical Parameter Estimation

```python
def estimate_material_properties(structure: Structure) -> tuple[float, float, float]:
    """
    Estimate lambda_ep, omega_log, and avg_mass from composition.
    
    Returns:
        (lambda_ep, omega_log, avg_mass)
    """
    composition = structure.composition
    avg_mass = composition.weight / composition.num_atoms
    
    # Base values
    lambda_ep = 0.3
    omega_log = 800.0 / np.sqrt(avg_mass / 10.0)
    
    # Boosts
    h_fraction = composition.get_atomic_fraction(Element("H"))
    lambda_ep += 1.5 * h_fraction  # Hydrides
    
    tm_elements = [Element(e) for e in ["Nb", "V", "Mo", "Ti", "Zr"]]
    for element in tm_elements:
        if element in composition:
            lambda_ep += 0.1 * composition.get_atomic_fraction(element)
    
    return float(lambda_ep), float(omega_log), float(avg_mass)
```

---

## Current Limitations & Next Steps

### Known Limitations (V1.0)

1. **Physics Accuracy**: 30% average (elements: 77%, compounds: 23%)
   - Root cause: Empirical parameter estimation too conservative
   - Solution: Tier 1 calibration (4-8 hours)

2. **Multi-Band Effects**: Not implemented
   - Affects: MgB₂, NbSe₂ (multi-band superconductors)
   - Solution: Tier 2 calibration (1-2 weeks)

3. **Cuprate Predictions**: 4% accuracy
   - Root cause: Wrong physics (BCS vs d-wave)
   - Solution: Tier 3 ML model (2-3 weeks)

4. **Database Persistence**: Not enabled
   - Predictions not saved to database yet
   - Solution: Update `/predict` endpoint (2 hours)

5. **Materials Project API**: Not configured
   - Using dummy structures instead of real ones
   - Solution: Set MP_API_KEY in structure_utils.py

### Tier 1 Calibration (4-8 hours)

**Goal**: 50% accuracy for BCS materials

**Tasks**:
1. Create `DEBYE_TEMP_DB` lookup table (20 materials)
2. Create `LAMBDA_CORRECTIONS` by material class
3. Update `estimate_material_properties()` to use lookups
4. Re-test 21-material benchmark suite

**Files to Modify**:
- `app/src/htc/structure_utils.py` (add lookups)
- `scripts/test_htc_materials.sh` (update expected values)

**Expected Improvement**:
```
Before: 30% average
After:  50% average (elements: 85%, A15: 60%, MgB2: 50%)
```

---

## Troubleshooting

### Prediction Always Returns Same Value

**Symptom**: All materials return 39 K (MgB₂ mock value)

**Diagnosis**:
```bash
# Check if structure_utils is being used
curl -X POST https://ard-backend-dydzexswua-uc.a.run.app/api/htc/predict \
  -H "Content-Type: application/json" \
  -d '{"composition": "Pb"}' | jq '.lambda_ep'

# If lambda_ep is always 0.7, structure_utils is not working
```

**Fix**: Verify pymatgen is installed in Docker image (`requirements.txt` includes `pymatgen==2024.3.1`)

### API Returns 501 "HTC module not available"

**Symptom**: Endpoint returns 501 error

**Diagnosis**:
```bash
# Check health endpoint
curl https://ard-backend-dydzexswua-uc.a.run.app/api/htc/health | jq

# Look for import errors in logs
gcloud run services logs tail ard-backend --region us-central1 | grep "Import"
```

**Fix**: Ensure all dependencies are in `pyproject.toml` under `[project.optional-dependencies.htc]`

### Database Migration Fails

**Symptom**: `KeyError: '002_add_bete_runs'` during `alembic upgrade head`

**Diagnosis**:
```bash
# Check migration history
cd app
alembic history
alembic current
```

**Fix**: Verify `down_revision` matches actual revision IDs in `app/alembic/versions/*.py`

### Cloud Build Upload Too Large

**Symptom**: `gcloud builds submit` hangs or times out

**Diagnosis**:
```bash
# Check upload size
du -sh app/
```

**Fix**: Update `app/.gcloudignore` to exclude large files:
```
venv/
node_modules/
*.pth
*.pt
*.onnx
```

---

## Monitoring & Logs

### Cloud Run Logs

```bash
# Tail logs
gcloud run services logs tail ard-backend --region us-central1

# Filter for HTC
gcloud run services logs read ard-backend --region us-central1 --filter "textPayload=~htc"

# Filter for errors
gcloud run services logs read ard-backend --region us-central1 --filter "severity=ERROR"
```

### Metrics

```bash
# Check request count (last hour)
gcloud monitoring time-series list \
  --filter 'metric.type="run.googleapis.com/request_count" AND resource.labels.service_name="ard-backend"' \
  --format json

# Check response time (last hour)
gcloud monitoring time-series list \
  --filter 'metric.type="run.googleapis.com/request_latencies" AND resource.labels.service_name="ard-backend"' \
  --format json
```

### Database Health

```bash
# Check database size
psql -h localhost -p 5433 -U ard_user -d ard_intelligence << 'EOF'
SELECT 
  pg_size_pretty(pg_database_size('ard_intelligence')) as database_size,
  pg_size_pretty(pg_total_relation_size('htc_predictions')) as htc_predictions_size;
EOF

# Check record counts
psql -h localhost -p 5433 -U ard_user -d ard_intelligence << 'EOF'
SELECT 
  (SELECT COUNT(*) FROM experiments) as experiments,
  (SELECT COUNT(*) FROM optimization_runs) as runs,
  (SELECT COUNT(*) FROM htc_predictions) as htc_predictions;
EOF
```

---

## Contact & Resources

**Project**: Periodic Labs - Autonomous R&D Intelligence Layer  
**Module**: HTC (High-Temperature Superconductor) Optimization Framework  
**Contact**: b@thegoatnote.com  
**License**: Apache 2.0

**Documentation**:
- [HTC Integration Guide](docs/HTC_INTEGRATION.md)
- [API Examples](docs/HTC_API_EXAMPLES.md)
- [Production Monitoring](docs/PRODUCTION_MONITORING.md)
- [Physics Calibration Status](HTC_PHYSICS_CALIBRATION_STATUS_OCT10_2025.md)
- [Production Milestone](HTC_PRODUCTION_MILESTONE_OCT10_2025.md)

**External Resources**:
- [Allen & Dynes (1975)](https://doi.org/10.1103/PhysRevB.12.905) - Original BCS formula
- [Materials Project](https://materialsproject.org/) - DFT database
- [SuperCon Database](https://supercon.nims.go.jp/index_en.html) - Experimental Tc values
- [Pymatgen Docs](https://pymatgen.org/) - Materials analysis library

---

**Last Updated**: October 10, 2025 @ 3:45 PM PST  
**Document Version**: 1.0  
**Status**: Production-deployed (ready for Tier 1 calibration)

