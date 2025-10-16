# Phase 3 Implementation Plan - Week 7 (Oct 13-20, 2025)
## Target: A+ Grade (4.0/4.0) - Cutting-Edge Research Contributions

**Based on Web Research: October 6, 2025**

---

## 🎯 Executive Summary

**Goal**: Transform A- scientific excellence into A+ publishable research  
**Timeline**: 12 weeks (Oct 13 - Dec 31, 2025)  
**Grade Target**: A+ (4.0/4.0)  
**Publication Target**: 4 top-tier conference papers (ICSE, ISSTA, SC, SIAM CSE)

**Week 7 Focus** (Oct 13-20):
1. ✅ Hermetic Builds with Nix Flakes
2. ✅ SLSA Level 3+ Attestation
3. ✅ ML-Powered Test Selection Foundation

---

## 📊 Current Assets (Phase 2 Complete)

### Infrastructure ✅
- **Python 3.12** with modern type hints
- **GitHub Actions CI/CD** with dual-job architecture (fast + chem)
- **Cloud Run** deployment with Cloud SQL integration
- **Docker** with BuildKit layer caching
- **uv** for fast dependency management + 3 lock files

### Testing ✅
- **28 tests** with 100% pass rate
- **pytest** with markers (`fast`, `chem`, `numerical`, `property`, `benchmark`)
- **Hypothesis** for property-based testing
- **pytest-benchmark** for continuous performance tracking
- **Cloud SQL** with 205 experiments, 20 runs, 100+ queries

### Security ✅
- **pip-audit** automated vulnerability scanning
- **Dependabot** automated dependency updates
- **Secret Manager** for credentials
- **Rate limiting** and security headers

### Data ✅
- **Cloud SQL PostgreSQL 15** with 205 experiments
- **Cloud Storage** for static assets
- **Vertex AI** integration (Flash + Pro models)
- **Cost tracking** in database

---

## 🔬 Phase 3 Actions (Based on 2025 Best Practices)

### Action 1: Hermetic Builds with Nix Flakes ⭐ PRIMARY

**Goal**: Achieve bit-identical builds reproducible to 2035  
**Time**: Week 7 (Oct 13-20)  
**SLSA Level**: Contributes to Level 3+

#### Latest Best Practices (Oct 2025)

**Source**: NixOS Discourse, filmil/bazel-nix-flakes GitHub

**Key Insights**:
1. **Flake-utils for Multi-Platform**: Simplifies cross-platform support (Linux/macOS)
2. **Hermetic Isolation**: No dependency on pre-existing Nix installation
3. **Provenance Tracking**: Essential for SLSA compliance
4. **Community Focus**: Nix community actively working on SLSA integration

#### Implementation Strategy

**Step 1: Set Up Nix Flakes**

Create `flake.nix` at repository root:

```nix
{
  description = "Autonomous R&D Intelligence Layer - Hermetic Builds";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    pyproject-nix.url = "github:nix-community/pyproject.nix";
  };

  outputs = { self, nixpkgs, flake-utils, pyproject-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = false; };  # Strict FOSS for reproducibility
        };
        
        python = pkgs.python312;
        
        # Parse pyproject.toml
        project = pyproject-nix.lib.project.loadPyproject {
          projectRoot = ./.;
        };
        
        # Core dependencies (no chemistry)
        corePythonEnv = python.withPackages (ps: with ps; [
          fastapi
          uvicorn
          pydantic
          sqlalchemy
          alembic
          pytest
          pytest-benchmark
          hypothesis
          ruff
          mypy
        ]);
        
        # Full environment including chemistry
        fullPythonEnv = python.withPackages (ps: with ps; [
          # Core (as above)
          fastapi uvicorn pydantic sqlalchemy alembic
          pytest pytest-benchmark hypothesis ruff mypy
          # Chemistry
          numpy scipy scikit-learn
          # Note: pyscf, rdkit may require custom derivations
        ]);
        
      in
      {
        # Default development shell (fast, no chemistry)
        devShells.default = pkgs.mkShell {
          buildInputs = [
            corePythonEnv
            pkgs.postgresql_15
            pkgs.google-cloud-sdk
          ];
          
          shellHook = ''
            echo "🔬 Autonomous R&D Intelligence Layer - Hermetic Dev Shell"
            echo "Python: ${python.version}"
            echo "PostgreSQL: ${pkgs.postgresql_15.version}"
            python --version
            pytest --version
          '';
        };
        
        # Full shell with chemistry dependencies
        devShells.full = pkgs.mkShell {
          buildInputs = [
            fullPythonEnv
            pkgs.postgresql_15
            pkgs.google-cloud-sdk
            pkgs.cmake
            pkgs.gfortran
            pkgs.blas
            pkgs.lapack
          ];
          
          shellHook = ''
            echo "🧪 Full Chemistry Environment"
            echo "Includes: NumPy, SciPy, scikit-learn"
          '';
        };
        
        # CI shell (optimized for GitHub Actions)
        devShells.ci = pkgs.mkShell {
          buildInputs = [
            corePythonEnv
            pkgs.git
            pkgs.docker
          ];
        };
        
        # Build the application (Docker-free hermetic build)
        packages.default = pkgs.stdenv.mkDerivation {
          name = "ard-backend";
          src = ./.;
          
          buildInputs = [ corePythonEnv ];
          
          buildPhase = ''
            # Run tests as part of build (ensures quality)
            pytest tests/ -m "not chem and not slow" -v
            
            # Type checking
            mypy app/src --ignore-missing-imports
            
            # Linting
            ruff check .
          '';
          
          installPhase = ''
            mkdir -p $out/bin $out/app
            cp -r app/* $out/app/
            
            # Create wrapper script
            cat > $out/bin/ard-backend << EOF
            #!${pkgs.bash}/bin/bash
            cd $out/app
            ${corePythonEnv}/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8080
            EOF
            chmod +x $out/bin/ard-backend
          '';
          
          # Provenance metadata (SLSA requirement)
          meta = with pkgs.lib; {
            description = "Autonomous R&D Intelligence Layer";
            homepage = "https://github.com/GOATnote-Inc/periodicdent42";
            license = licenses.mit;
            platforms = platforms.linux ++ platforms.darwin;
            maintainers = [ "kiteboard" ];
          };
        };
        
        # Docker image built hermetically
        packages.docker = pkgs.dockerTools.buildLayeredImage {
          name = "ard-backend";
          tag = "hermetic-${self.rev or "dev"}";
          
          contents = [ self.packages.${system}.default ];
          
          config = {
            Cmd = [ "${self.packages.${system}.default}/bin/ard-backend" ];
            ExposedPorts = { "8080/tcp" = {}; };
            Env = [
              "PYTHONUNBUFFERED=1"
              "PROJECT_ID=periodicdent42"
            ];
          };
        };
        
        # Checks (run with `nix flake check`)
        checks = {
          tests = pkgs.runCommand "run-tests" {
            buildInputs = [ corePythonEnv ];
          } ''
            cp -r ${./.} source
            cd source
            pytest tests/ -m "not chem and not slow" -v
            touch $out
          '';
          
          lint = pkgs.runCommand "run-lint" {
            buildInputs = [ corePythonEnv pkgs.ruff ];
          } ''
            cp -r ${./.} source
            cd source
            ruff check .
            touch $out
          '';
          
          types = pkgs.runCommand "run-mypy" {
            buildInputs = [ corePythonEnv pkgs.mypy ];
          } ''
            cp -r ${./.} source
            cd source
            mypy app/src --ignore-missing-imports
            touch $out
          '';
        };
      }
    );
}
```

**Step 2: GitHub Actions Integration**

Update `.github/workflows/ci.yml`:

```yaml
name: CI with Nix Flakes

on:
  push:
    branches: [main]
  pull_request:

jobs:
  nix-hermetic:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Nix
        uses: DeterminateSystems/nix-installer-action@v9
      
      - name: Configure Nix Cache
        uses: DeterminateSystems/magic-nix-cache-action@v3
      
      - name: Verify flake
        run: nix flake check -L
      
      - name: Run hermetic tests
        run: nix develop .#ci --command pytest tests/ -m "not chem and not slow" -v
      
      - name: Build hermetically
        run: nix build .#default -L
      
      - name: Generate SBOM
        run: nix run nixpkgs#sbom-tool generate -b . -o sbom.json
      
      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: sbom.json
  
  nix-docker:
    runs-on: ubuntu-latest
    needs: nix-hermetic
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Nix
        uses: DeterminateSystems/nix-installer-action@v9
      
      - name: Build Docker image hermetically
        run: |
          nix build .#docker
          docker load < result
      
      - name: Test Docker image
        run: |
          docker run -d -p 8080:8080 --name test-container ard-backend:hermetic-${{ github.sha }}
          sleep 5
          curl http://localhost:8080/health | jq .
          docker stop test-container
```

**Success Metrics**:
- ✅ Bit-identical builds on different machines
- ✅ No dependency on system Python or pip
- ✅ Builds work on Linux and macOS (CI matrix)
- ✅ SBOM automatically generated
- ✅ Build time < 2 minutes (with Nix cache)

---

### Action 2: SLSA Level 3+ Attestation ⭐ PRIMARY

**Goal**: Cryptographic build provenance for supply chain security  
**Time**: Week 7 (Oct 13-20)  
**Standard**: SLSA v1.0 (latest as of Oct 2025)

#### Latest Best Practices (Oct 2025)

**Sources**: Sigstore, in-toto, SLSA v1.0 spec

**Key Tools**:
1. **Sigstore** - Keyless signing with transparency log
2. **in-toto** - Supply chain metadata framework
3. **GitHub Attestations** - Native SLSA support in Actions
4. **SLSA Verifier** - Automated verification

#### Implementation Strategy

**Step 1: GitHub SLSA Provenance**

Update `.github/workflows/cicd.yaml`:

```yaml
name: SLSA Build and Deploy

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  id-token: write  # Required for Sigstore
  packages: write
  attestations: write  # GitHub Attestations

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for provenance
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Build and push
        id: build
        uses: docker/build-push-action@v5
        with:
          context: app
          push: true
          tags: |
            gcr.io/periodicdent42/ard-backend:${{ github.sha }}
            gcr.io/periodicdent42/ard-backend:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true  # Automatic provenance generation
          sbom: true  # Automatic SBOM generation
      
      - name: Generate SLSA Provenance
        uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v1.10.0
        with:
          image: gcr.io/periodicdent42/ard-backend
          digest: ${{ steps.build.outputs.digest }}
      
      - name: Sign with Sigstore
        uses: sigstore/gh-action-sigstore-python@v2.1.1
        with:
          inputs: ./app/requirements.lock
          upload-signing-artifacts: true
      
      - name: Generate in-toto attestation
        run: |
          # Install in-toto
          pip install in-toto
          
          # Create link metadata
          in-toto-run \
            --step-name build \
            --products gcr.io/periodicdent42/ard-backend:${{ github.sha }} \
            --key ${{ secrets.IN_TOTO_KEY }} \
            -- docker build app
      
      - name: Upload attestations
        uses: actions/attest-build-provenance@v1
        with:
          subject-name: gcr.io/periodicdent42/ard-backend
          subject-digest: ${{ steps.build.outputs.digest }}
          push-to-registry: true
  
  verify:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Install SLSA Verifier
        run: |
          wget https://github.com/slsa-framework/slsa-verifier/releases/latest/download/slsa-verifier-linux-amd64
          chmod +x slsa-verifier-linux-amd64
          sudo mv slsa-verifier-linux-amd64 /usr/local/bin/slsa-verifier
      
      - name: Verify SLSA Provenance
        run: |
          slsa-verifier verify-image \
            gcr.io/periodicdent42/ard-backend:${{ github.sha }} \
            --source-uri github.com/GOATnote-Inc/periodicdent42 \
            --source-branch main
      
      - name: Verify Sigstore signature
        uses: sigstore/cosign-installer@v3.4.0
      
      - run: |
          cosign verify \
            --certificate-identity-regexp=https://github.com/GOATnote-Inc/periodicdent42 \
            --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
            gcr.io/periodicdent42/ard-backend:${{ github.sha }}
  
  deploy:
    runs-on: ubuntu-latest
    needs: verify
    steps:
      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: ard-backend
          image: gcr.io/periodicdent42/ard-backend:${{ github.sha }}
          region: us-central1
          flags: |
            --cpu=2
            --memory=4Gi
            --min-instances=1
            --max-instances=10
            --set-env-vars=GCP_SQL_INSTANCE=periodicdent42:us-central1:ard-intelligence-db
            --update-secrets=DB_PASSWORD=db-password:latest
```

**Step 2: SLSA Verification in Deployment**

Create `scripts/verify_slsa.sh`:

```bash
#!/bin/bash
set -euo pipefail

IMAGE="$1"

echo "🔒 Verifying SLSA Level 3 compliance for: $IMAGE"

# 1. Verify SLSA provenance
echo "Step 1: Verifying SLSA provenance..."
slsa-verifier verify-image "$IMAGE" \
  --source-uri github.com/GOATnote-Inc/periodicdent42 \
  --source-branch main \
  --print-provenance > provenance.json

# 2. Verify Sigstore signature
echo "Step 2: Verifying Sigstore signature..."
cosign verify \
  --certificate-identity-regexp=https://github.com/GOATnote-Inc/periodicdent42 \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  "$IMAGE"

# 3. Extract and verify SBOM
echo "Step 3: Extracting SBOM..."
cosign download sbom "$IMAGE" > sbom.json
echo "SBOM downloaded: $(jq '.metadata.component.name' sbom.json)"

# 4. Check for vulnerabilities in SBOM
echo "Step 4: Scanning SBOM for vulnerabilities..."
grype sbom:sbom.json --fail-on=critical

echo "✅ SLSA Level 3 verification complete!"
```

**Success Metrics**:
- ✅ SLSA Level 3+ compliance verified
- ✅ All builds have cryptographic attestations
- ✅ Provenance traceable to GitHub commit
- ✅ Sigstore signatures on all artifacts
- ✅ SBOM automatically generated and verified
- ✅ Zero critical vulnerabilities in production

---

### Action 3: ML-Powered Test Selection 🤖 RESEARCH

**Goal**: 70% CI time reduction through intelligent test prioritization  
**Time**: Weeks 7-10 (Oct 13 - Nov 10)  
**Publication Target**: ISSTA 2026

#### Latest Best Practices (Oct 2025)

**Source**: TestImpact.ai, academic research, industry ML/CI practices

**Key Approaches**:
1. **Feature Engineering**: Code complexity, change patterns, historical failures
2. **Model Types**: Random Forest, XGBoost, Neural Networks
3. **Real-time Prediction**: Integrate with CI pipeline
4. **Continuous Learning**: Retrain on new test results

#### Implementation Strategy

**Step 1: Data Collection Infrastructure**

Create `app/src/services/test_telemetry.py`:

```python
"""Test telemetry collection for ML-powered test selection."""

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

from sqlalchemy import Column, String, Float, Integer, Boolean, JSON, DateTime
from sqlalchemy.sql import func

from .db import Base, get_session


@dataclass
class TestExecution:
    """Single test execution record."""
    test_name: str
    duration_ms: float
    passed: bool
    commit_sha: str
    branch: str
    changed_files: List[str]
    test_file: str
    timestamp: str
    
    # Code change features
    lines_added: int
    lines_deleted: int
    files_changed: int
    complexity_delta: float
    
    # Historical features
    recent_failure_rate: float
    avg_duration: float
    days_since_last_change: int


class TestTelemetry(Base):
    """Database model for test execution telemetry."""
    __tablename__ = "test_telemetry"
    
    id = Column(String, primary_key=True)
    test_name = Column(String, nullable=False, index=True)
    duration_ms = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False)
    commit_sha = Column(String, nullable=False, index=True)
    branch = Column(String, nullable=False)
    changed_files = Column(JSON, nullable=False)
    test_file = Column(String, nullable=False)
    
    # Features for ML
    lines_added = Column(Integer, default=0)
    lines_deleted = Column(Integer, default=0)
    files_changed = Column(Integer, default=0)
    complexity_delta = Column(Float, default=0.0)
    recent_failure_rate = Column(Float, default=0.0)
    avg_duration = Column(Float, default=0.0)
    days_since_last_change = Column(Integer, default=0)
    
    created_at = Column(DateTime, server_default=func.now())


class TestCollector:
    """Collects test execution data for ML training."""
    
    def __init__(self):
        self.session = get_session()
    
    def collect_test_result(self, execution: TestExecution) -> None:
        """Store test execution result in database."""
        record = TestTelemetry(
            id=self._generate_id(execution),
            **asdict(execution)
        )
        self.session.add(record)
        self.session.commit()
    
    def get_changed_files(self, commit_sha: str) -> List[str]:
        """Get files changed in commit using git."""
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_sha],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split("\n")
    
    def calculate_complexity_delta(self, file_path: str, commit_sha: str) -> float:
        """Calculate cyclomatic complexity change for a file."""
        # Use radon to calculate complexity
        try:
            result = subprocess.run(
                ["radon", "cc", file_path, "-a"],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse average complexity from output
            for line in result.stdout.split("\n"):
                if "Average complexity:" in line:
                    return float(line.split(":")[-1].strip()[0])
            return 0.0
        except Exception:
            return 0.0
    
    def get_recent_failure_rate(self, test_name: str, days: int = 30) -> float:
        """Calculate recent failure rate for a test."""
        from sqlalchemy import and_, func as sql_func
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        total = self.session.query(TestTelemetry).filter(
            and_(
                TestTelemetry.test_name == test_name,
                TestTelemetry.created_at >= cutoff
            )
        ).count()
        
        if total == 0:
            return 0.0
        
        failures = self.session.query(TestTelemetry).filter(
            and_(
                TestTelemetry.test_name == test_name,
                TestTelemetry.created_at >= cutoff,
                TestTelemetry.passed == False
            )
        ).count()
        
        return failures / total
    
    def export_training_data(self, output_path: Path) -> None:
        """Export all test telemetry for ML training."""
        records = self.session.query(TestTelemetry).all()
        
        data = [
            {
                "test_name": r.test_name,
                "duration_ms": r.duration_ms,
                "passed": r.passed,
                "features": {
                    "lines_added": r.lines_added,
                    "lines_deleted": r.lines_deleted,
                    "files_changed": r.files_changed,
                    "complexity_delta": r.complexity_delta,
                    "recent_failure_rate": r.recent_failure_rate,
                    "avg_duration": r.avg_duration,
                    "days_since_last_change": r.days_since_last_change,
                }
            }
            for r in records
        ]
        
        output_path.write_text(json.dumps(data, indent=2))
        print(f"✅ Exported {len(data)} test records to {output_path}")
    
    @staticmethod
    def _generate_id(execution: TestExecution) -> str:
        """Generate unique ID for test execution."""
        key = f"{execution.test_name}:{execution.commit_sha}:{execution.timestamp}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
```

**Step 2: pytest Plugin for Automatic Collection**

Create `app/tests/conftest.py` enhancement:

```python
"""pytest configuration with ML telemetry collection."""

import pytest
import time
from pathlib import Path

from app.src.services.test_telemetry import TestCollector, TestExecution


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Collect test execution data after each test."""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":  # Only collect for actual test execution
        collector = TestCollector()
        
        # Get changed files from environment (set by CI)
        import os
        commit_sha = os.getenv("GITHUB_SHA", "local")
        branch = os.getenv("GITHUB_REF_NAME", "local")
        
        changed_files = []
        if commit_sha != "local":
            changed_files = collector.get_changed_files(commit_sha)
        
        # Calculate features
        test_file = str(Path(item.fspath).relative_to(Path.cwd()))
        lines_added = len([f for f in changed_files if f.endswith(".py")])  # Simplified
        
        execution = TestExecution(
            test_name=item.nodeid,
            duration_ms=report.duration * 1000,
            passed=report.passed,
            commit_sha=commit_sha,
            branch=branch,
            changed_files=changed_files,
            test_file=test_file,
            timestamp=time.time(),
            lines_added=lines_added,
            lines_deleted=0,  # Calculate from git diff
            files_changed=len(changed_files),
            complexity_delta=0.0,  # Calculate if needed
            recent_failure_rate=collector.get_recent_failure_rate(item.nodeid),
            avg_duration=0.0,  # Calculate from history
            days_since_last_change=0,
        )
        
        collector.collect_test_result(execution)
```

**Step 3: ML Model Training**

Create `scripts/train_test_selector.py`:

```python
"""Train ML model for test selection."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, precision_recall_curve
import joblib

from app.src.services.test_telemetry import TestCollector


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare training data."""
    with open(data_path) as f:
        data = json.load(f)
    
    rows = []
    for record in data:
        row = {
            "test_name": record["test_name"],
            "failed": not record["passed"],  # Target variable
            **record["features"]
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def train_model(df: pd.DataFrame):
    """Train test selection model."""
    # Features and target
    feature_cols = [
        "lines_added", "lines_deleted", "files_changed",
        "complexity_delta", "recent_failure_rate",
        "avg_duration", "days_since_last_change"
    ]
    
    X = df[feature_cols]
    y = df["failed"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Try multiple models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n🔬 Training {name}...")
        model.fit(X_train, y_train)
        
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        print(f"  CV F1 Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        # Test set evaluation
        y_pred = model.predict(X_test)
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Pass", "Fail"]))
        
        # Feature importance
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            print(f"\n  Top 3 Features:")
            print(importance.head(3).to_string(index=False))
        
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = model
    
    return best_model, X_test, y_test


def analyze_time_savings(model, df: pd.DataFrame):
    """Estimate CI time savings from test selection."""
    feature_cols = [
        "lines_added", "lines_deleted", "files_changed",
        "complexity_delta", "recent_failure_rate",
        "avg_duration", "days_since_last_change"
    ]
    
    X = df[feature_cols]
    
    # Predict failure probability
    failure_probs = model.predict_proba(X)[:, 1]
    
    # Sort tests by failure probability (run high-prob first)
    df["failure_prob"] = failure_probs
    df_sorted = df.sort_values("failure_prob", ascending=False)
    
    # Calculate cumulative time if we run tests in priority order
    total_time = df["avg_duration"].sum()
    
    # Assume we stop after catching 95% of failures
    failures = df[df["failed"] == True]
    target_failures = int(len(failures) * 0.95)
    
    tests_to_run = df_sorted.head(target_failures)
    time_used = tests_to_run["avg_duration"].sum()
    
    time_saved = total_time - time_used
    savings_pct = (time_saved / total_time) * 100
    
    print(f"\n📊 Time Savings Analysis:")
    print(f"  Total test time: {total_time/1000:.1f}s")
    print(f"  Time with ML selection: {time_used/1000:.1f}s")
    print(f"  Time saved: {time_saved/1000:.1f}s ({savings_pct:.1f}%)")
    print(f"  Tests run: {len(tests_to_run)}/{len(df)} ({len(tests_to_run)/len(df)*100:.1f}%)")


def main():
    """Main training pipeline."""
    print("🤖 ML-Powered Test Selection - Training Pipeline\n")
    
    # 1. Export telemetry data
    print("Step 1: Exporting test telemetry...")
    collector = TestCollector()
    data_path = Path("test_telemetry.json")
    collector.export_training_data(data_path)
    
    # 2. Load and prepare data
    print("\nStep 2: Loading training data...")
    df = load_training_data(data_path)
    print(f"  Loaded {len(df)} test executions")
    print(f"  Failure rate: {df['failed'].mean()*100:.1f}%")
    
    # 3. Train model
    print("\nStep 3: Training models...")
    model, X_test, y_test = train_model(df)
    
    # 4. Analyze savings
    print("\nStep 4: Analyzing time savings...")
    analyze_time_savings(model, df)
    
    # 5. Save model
    model_path = Path("models/test_selector.pkl")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")


if __name__ == "__main__":
    main()
```

**Step 4: CI Integration**

Update `.github/workflows/ci.yml`:

```yaml
jobs:
  ml-test-selection:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2  # Need previous commit for diff
      
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      
      - name: Install dependencies
        run: |
          pip install scikit-learn joblib pandas numpy
      
      - name: Load ML model
        run: |
          # Download trained model from GCS or artifacts
          gsutil cp gs://periodicdent42-models/test_selector.pkl models/
      
      - name: Select tests to run
        id: select
        run: |
          python scripts/predict_tests.py \
            --changed-files=$(git diff --name-only HEAD^ HEAD) \
            --output=selected_tests.txt
      
      - name: Run selected tests
        run: |
          pytest $(cat selected_tests.txt) -v --tb=short
      
      - name: Collect telemetry (always run)
        if: always()
        env:
          GITHUB_SHA: ${{ github.sha }}
          GITHUB_REF_NAME: ${{ github.ref_name }}
        run: |
          # Telemetry collection happens via pytest plugin
          echo "✅ Test telemetry collected"
```

**Success Metrics**:
- ✅ 70% CI time reduction (target)
- ✅ 95% failure detection rate
- ✅ < 5% false negatives (missed failures)
- ✅ Model F1 score > 0.80
- ✅ Continuous learning (weekly retraining)

---

## 📅 Week 7 Implementation Timeline

### Day 1-2 (Oct 13-14): Nix Flakes Setup
- [ ] Install Nix on development machine
- [ ] Create `flake.nix` with basic structure
- [ ] Test `nix develop` shell
- [ ] Verify reproducibility (same hash on different machines)

### Day 3-4 (Oct 15-16): SLSA Attestation
- [ ] Set up Sigstore in GitHub Actions
- [ ] Configure SLSA provenance generation
- [ ] Add in-toto attestation
- [ ] Create verification script

### Day 5-7 (Oct 17-19): ML Test Selection Foundation
- [ ] Add `test_telemetry` table to database
- [ ] Implement pytest plugin for collection
- [ ] Run tests with collection enabled
- [ ] Export training data
- [ ] Train initial model

### Day 8 (Oct 20): Verification & Documentation
- [ ] Run full hermetic build
- [ ] Verify SLSA Level 3 compliance
- [ ] Document all implementations
- [ ] Update agents.md with progress

---

## 📊 Success Metrics for Week 7

### Hermetic Builds ✅
- [ ] Bit-identical builds on Linux and macOS
- [ ] Build time < 2 minutes (with cache)
- [ ] No system dependencies (Python, pip, etc.)
- [ ] SBOM automatically generated
- [ ] Works offline (after initial download)

### SLSA Attestation ✅
- [ ] SLSA Level 3 verified
- [ ] All artifacts signed with Sigstore
- [ ] Provenance traceable to Git commit
- [ ] Verification script works
- [ ] Documentation complete

### ML Test Selection ✅
- [ ] 100+ test executions collected
- [ ] Training pipeline runs successfully
- [ ] Model F1 score > 0.60 (initial)
- [ ] Feature importance identified
- [ ] Time savings analysis complete

---

## 🎓 Publication Strategy

### Paper 1: ICSE 2026 (Submission: Aug 2026)
**Title**: "Hermetic Builds for Scientific Reproducibility: A Nix-Based Approach"

**Contributions**:
1. Novel integration of Nix flakes with Python scientific computing
2. Case study: 205 experiments with bit-identical results
3. Comparison with Docker, Conda, and traditional pip
4. 10-year reproducibility validation (2025-2035)

### Paper 2: ISSTA 2026 (Submission: Feb 2026)
**Title**: "ML-Powered Test Selection in Research Codebases"

**Contributions**:
1. Novel features for scientific code (complexity, domain-specific metrics)
2. 70% CI time reduction with 95% failure detection
3. Comparison with existing methods (e.g., TestImpact)
4. Open-source dataset of 10,000+ test executions

### Paper 3: SC'26 (Submission: May 2026)
**Title**: "Chaos Engineering for Computational Science Workflows"

**Contributions**:
1. Systematic failure injection for scientific pipelines
2. 10% failure resilience validation
3. Case studies: RL optimization, Bayesian optimization
4. Open-source chaos testing framework

### Paper 4: SIAM CSE 2027 (Submission: Oct 2026)
**Title**: "Continuous Benchmarking Best Practices for Scientific Computing"

**Contributions**:
1. Performance regression detection at machine precision
2. Property-based testing for scientific code
3. 5-year performance tracking case study
4. Recommendations for computational science CI/CD

---

## 🚀 Phase 3 Complete Roadmap

### Weeks 7-8 (Oct 13-27): Foundation
- ✅ Hermetic builds (Nix flakes)
- ✅ SLSA Level 3+ attestation
- ✅ ML test selection foundation
- ✅ Test telemetry collection

### Weeks 9-10 (Oct 28 - Nov 10): ML Optimization
- [ ] Train production ML model (1000+ executions)
- [ ] Deploy ML test selection in CI
- [ ] Achieve 70% time reduction
- [ ] Continuous learning pipeline

### Weeks 11-12 (Nov 11-24): Chaos Engineering
- [ ] Failure injection framework
- [ ] Random hardware failures (simulated)
- [ ] Network latency injection
- [ ] Database failure simulation
- [ ] 10% failure resilience validation

### Weeks 13-14 (Nov 25 - Dec 8): Result Regression
- [ ] Numerical result tracking in database
- [ ] Automatic comparison against baselines
- [ ] Alert on > 1e-10 regression
- [ ] Visualization dashboard

### Weeks 15-16 (Dec 9-22): Profiling & DVC
- [ ] Continuous profiling (py-spy, scalene)
- [ ] Flamegraph generation in CI
- [ ] DVC setup with Cloud Storage
- [ ] Data versioning for experiments

### Week 17 (Dec 23-29): Final Integration
- [ ] All Phase 3 features integrated
- [ ] Full documentation
- [ ] Paper drafts started
- [ ] A+ verification

---

## 💰 Cost Estimate

### Cloud Infrastructure
- **Cloud Run**: $10-20/month (same as Phase 2)
- **Cloud SQL**: $30-50/month (same as Phase 2)
- **Cloud Storage**: $5-10/month (DVC data)
- **Vertex AI**: Variable (usage-based)

### CI/CD
- **GitHub Actions**: Free (public repo)
- **Nix Cache**: Free (Determinate Systems)
- **Sigstore**: Free (community service)

### Total: ~$50-80/month (minimal increase from Phase 2)

---

## 🎯 Grade Rubric

### A+ Criteria (4.0/4.0)
- [x] ✅ All A- criteria met (Phase 2)
- [ ] 🔄 Hermetic builds (bit-identical)
- [ ] 🔄 SLSA Level 3+ attestation
- [ ] 🔄 ML test selection (70% reduction)
- [ ] ⏳ Chaos engineering (10% resilience)
- [ ] ⏳ Result regression detection
- [ ] ⏳ 4 conference paper drafts
- [ ] ⏳ Continuous profiling
- [ ] ⏳ DVC data versioning

### Current Status
- **Phase 1**: B+ (3.3/4.0) ✅ Complete
- **Phase 2**: A- (3.7/4.0) ✅ Complete
- **Phase 3**: In Progress (Week 7 starting Oct 13)

---

## 🏁 Next Steps (Immediate)

### This Week (Oct 6-12): Preparation
1. **Read Nix documentation** (2 hours)
2. **Review SLSA v1.0 spec** (1 hour)
3. **Study ML test selection papers** (2 hours)
4. **Set up local Nix environment** (1 hour)
5. **Plan Week 7 implementation** (30 min)

### Week 7 (Oct 13-20): Implementation
1. **Monday-Tuesday**: Nix flakes setup
2. **Wednesday-Thursday**: SLSA attestation
3. **Friday-Sunday**: ML test selection
4. **Monday (Oct 20)**: Documentation & verification

---

## 📚 References (October 2025)

### Nix & Hermetic Builds
- NixOS Discourse: SLSA integration discussions
- GitHub: filmil/bazel-nix-flakes (hermetic example)
- pyproject-nix: Python package management in Nix

### SLSA & Supply Chain
- SLSA v1.0 Specification
- Sigstore documentation
- in-toto framework
- GitHub Attestations

### ML Test Selection
- TestImpact.ai documentation
- Academic papers on test prioritization
- Scikit-learn best practices

---

**Status**: READY TO BEGIN WEEK 7 (Oct 13, 2025)  
**Target**: A+ Grade (4.0/4.0) by Dec 31, 2025  
**Publication**: 4 papers submitted to top-tier venues in 2026

🎓 Let's achieve publishable research excellence! 🚀
