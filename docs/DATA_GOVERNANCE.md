# Data Governance & Retention Policy

**Autonomous R&D Intelligence Layer**  
*GOATnote Research Lab Initiative*

---

## Overview

This document outlines data retention policies, PII handling guidelines, and compliance requirements for the Epistemic CI system.

---

## Data Assets

### 1. CI Run Telemetry (`data/ci_runs.jsonl`)

**Classification:** Operational Data  
**Retention:** 12 months rolling  
**PII Risk:** Low  
**Storage:** DVC + GCS (`gs://periodicdent42-data`)

**Contents:**
- Test execution times, costs, failure rates
- Git commit SHAs, branch names, file change counts
- ML model predictions (failure probability)
- NO usernames, emails, IP addresses

**Retention Policy:**
- Keep all data for 12 months
- After 12 months: Archive to cold storage (GCS Nearline/Coldline)
- After 24 months: Delete unless flagged for research

**Checksum Validation:**
```bash
make data-check  # Validate before each training run
```

---

### 2. ML Models (`models/selector-v*.pkl`)

**Classification:** Model Artifacts  
**Retention:** Indefinite (version-controlled)  
**PII Risk:** None  
**Storage:** DVC + GCS

**Contents:**
- Serialized GradientBoostingClassifier
- Feature weights, hyperparameters
- Training metadata (F1, precision, recall)

**Retention Policy:**
- Keep all model versions indefinitely
- Tag production models: `models/selector-prod.pkl`
- Deprecate models >6 months old (move to `models/archive/`)

**Versioning:**
```bash
dvc add models/selector-v1.pkl
git add models/selector-v1.pkl.dvc
git commit -m "Train selector-v1 on 200 CI runs"
```

---

### 3. Experiment Ledgers (`experiments/ledger/*.json`)

**Classification:** Research Data  
**Retention:** Indefinite (for publication)  
**PII Risk:** None  
**Storage:** Git + DVC

**Contents:**
- Per-run epistemic metrics (EIG, detection rate, efficiency)
- Test selection decisions
- Reproducibility metadata (seed, commit SHA, env hash)

**Retention Policy:**
- Keep indefinitely for research reproducibility
- Required for PhD thesis, ICSE/ISSTA/SC papers
- Backup to cold storage after publication

**Schema Validation:**
```bash
jsonschema -i experiments/ledger/abc123def456.json schemas/experiment_ledger.schema.json
```

---

### 4. Reproducibility Artifacts (`artifact/`)

**Classification:** Ephemeral CI Outputs  
**Retention:** 90 days  
**PII Risk:** None  
**Storage:** GitHub Actions Artifacts (auto-deleted)

**Contents:**
- CI reports (`ci_report.md`, `ci_metrics.json`)
- Test selection lists (`selected_tests.json`, `eig_rankings.json`)
- Build logs, reproducibility appendices

**Retention Policy:**
- GitHub Actions: Auto-delete after 90 days
- Production runs: Upload to GCS for 12 months
- Archive key artifacts for research

---

## PII Handling

### What is PII?

Personally Identifiable Information (PII) includes:
- Names, email addresses, phone numbers
- IP addresses, device identifiers
- Usernames (if real names), SSH keys
- Any data that identifies an individual

### PII Policy

**This system processes NO PII.**

If PII is accidentally committed:
1. **Immediate action:** Contact b@thegoatnote.com
2. **Rotation:** Rotate all affected credentials
3. **Purge:** Remove from Git history (`git filter-repo`)
4. **Audit:** Run secrets scan (`make secrets-scan`)

### PII Audit Checklist

- [ ] No names/emails in test data
- [ ] No IP addresses in logs
- [ ] No API keys in .env files (use Secret Manager)
- [ ] No usernames in commit messages
- [ ] No SSH keys in repo

---

## DVC Configuration

### Setup

```bash
# Install DVC with Google Cloud Storage support
pip install 'dvc[gs]'

# Initialize DVC
make data-init

# Configure credentials
gcloud auth application-default login
```

### Daily Workflow

```bash
# Pull latest data before training
make data-pull

# Train model
make train

# Push model to remote
make data-push
```

### Retention Configuration

Set lifecycle rules in GCS:

```bash
# Archive CI runs after 12 months
gsutil lifecycle set lifecycle.json gs://periodicdent42-data
```

`lifecycle.json`:
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 365, "matchesPrefix": ["data/ci_runs"]}
      },
      {
        "action": {"type": "Delete"},
        "condition": {"age": 730, "matchesPrefix": ["artifact/"]}
      }
    ]
  }
}
```

---

## Compliance

### GDPR / CCPA

**Status:** Not applicable (no PII collected)

If PII is collected in future:
- Implement data subject access requests (DSAR)
- Add deletion endpoints
- Document legal basis (consent, legitimate interest)
- Appoint Data Protection Officer (DPO)

### HIPAA

**Status:** Not applicable (no healthcare data)

### Export Control (ITAR/EAR)

**Status:** Research data only (no export restrictions)

If integrating with defense/aerospace:
- Review EAR 99 classification
- Implement access controls for non-US citizens
- Document technology transfer agreements

---

## Backup & Recovery

### Backup Schedule

- **Daily:** Automated DVC push to GCS (versioned)
- **Weekly:** Snapshot to cold storage (GCS Archive)
- **Monthly:** Offsite backup to separate GCP project

### Recovery Procedures

**Scenario 1: Accidental data deletion**

```bash
# Restore from DVC
dvc pull
```

**Scenario 2: Corrupted models**

```bash
# Rollback to previous version
git checkout HEAD~1 models/selector-v1.pkl.dvc
dvc checkout models/selector-v1.pkl.dvc
```

**Scenario 3: Total data loss**

```bash
# Restore from GCS
gsutil -m rsync -r gs://periodicdent42-data/ ./data/
```

---

## Access Controls

### GCS Bucket Permissions

- **Read-only:** CI service account (`ard-ci@periodicdent42.iam.gserviceaccount.com`)
- **Read-write:** Developers (`b@thegoatnote.com`)
- **Admin:** Project owner only

### DVC Remote Access

```bash
# Grant read access
gcloud storage buckets add-iam-policy-binding gs://periodicdent42-data \
  --member=serviceAccount:ard-ci@periodicdent42.iam.gserviceaccount.com \
  --role=roles/storage.objectViewer

# Grant write access (developers only)
gcloud storage buckets add-iam-policy-binding gs://periodicdent42-data \
  --member=user:b@thegoatnote.com \
  --role=roles/storage.objectAdmin
```

---

## Audit Log

All data access is logged via Google Cloud Audit Logs:

```bash
# View recent data access
gcloud logging read "resource.type=gcs_bucket AND resource.labels.bucket_name=periodicdent42-data" \
  --limit 50 --format json
```

---

## Contact

**Data Governance Questions:**  
Email: b@thegoatnote.com  
Subject: "Data Governance - [Your Question]"

**Security Incidents:**  
Email: b@thegoatnote.com  
Subject: "SECURITY INCIDENT - [Brief Description]"

**Last Updated:** October 7, 2025  
**Review Schedule:** Quarterly (every 90 days)  
**Next Review:** January 7, 2026
