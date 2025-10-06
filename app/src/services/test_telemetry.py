"""Test telemetry collection for ML-powered test selection.

This module collects test execution data to train an ML model that can predict
which tests are most likely to fail given a set of code changes, enabling
intelligent test selection and 70% CI time reduction.

Phase 3 Week 7 Day 5-6: ML Test Selection Foundation
"""

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import Column, String, Float, Integer, Boolean, JSON, DateTime, Index
from sqlalchemy import and_, func as sql_func
from sqlalchemy.orm import Session

from .db import Base, get_session


@dataclass
class TestExecution:
    """Single test execution record with features for ML model."""
    
    # Test identification
    test_name: str
    test_file: str
    
    # Execution results
    duration_ms: float
    passed: bool
    error_message: Optional[str] = None
    
    # Source control context
    commit_sha: str
    branch: str
    changed_files: List[str] = None
    
    # Code change features (for ML)
    lines_added: int = 0
    lines_deleted: int = 0
    files_changed: int = 0
    complexity_delta: float = 0.0
    
    # Historical features (computed)
    recent_failure_rate: float = 0.0
    avg_duration: float = 0.0
    days_since_last_change: int = 0
    
    # Timestamp
    timestamp: float = None
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.changed_files is None:
            self.changed_files = []
        if self.timestamp is None:
            self.timestamp = time.time()


class TestTelemetry(Base):
    """Database model for test execution telemetry."""
    __tablename__ = "test_telemetry"
    
    # Primary key
    id = Column(String(16), primary_key=True)
    
    # Test identification
    test_name = Column(String(500), nullable=False, index=True)
    test_file = Column(String(500), nullable=False)
    
    # Execution results
    duration_ms = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False, index=True)
    error_message = Column(String, nullable=True)
    
    # Source control context
    commit_sha = Column(String(40), nullable=False, index=True)
    branch = Column(String(200), nullable=False)
    changed_files = Column(JSON, nullable=False)
    
    # Features for ML
    lines_added = Column(Integer, default=0)
    lines_deleted = Column(Integer, default=0)
    files_changed = Column(Integer, default=0)
    complexity_delta = Column(Float, default=0.0)
    recent_failure_rate = Column(Float, default=0.0)
    avg_duration = Column(Float, default=0.0)
    days_since_last_change = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, server_default=sql_func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_test_telemetry_test_name_created', 'test_name', 'created_at'),
        Index('idx_test_telemetry_passed_created', 'passed', 'created_at'),
    )


class TestCollector:
    """Collects test execution data for ML training."""
    
    def __init__(self, session: Optional[Session] = None):
        """Initialize collector with database session.
        
        Args:
            session: Optional SQLAlchemy session. If None, will create one.
        """
        self.session = session or get_session()
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure test_telemetry table exists."""
        try:
            # Try a simple query to check if table exists
            self.session.query(TestTelemetry).limit(1).all()
        except Exception as e:
            print(f"⚠️  test_telemetry table not found: {e}")
            print("   Run: alembic upgrade head")
    
    def collect_test_result(self, execution: TestExecution) -> None:
        """Store test execution result in database.
        
        Args:
            execution: TestExecution dataclass with test results and features
        """
        try:
            # Compute historical features
            execution.recent_failure_rate = self.get_recent_failure_rate(
                execution.test_name
            )
            execution.avg_duration = self.get_avg_duration(execution.test_name)
            execution.days_since_last_change = self.get_days_since_last_change(
                execution.test_file
            )
            
            # Create database record
            record = TestTelemetry(
                id=self._generate_id(execution),
                **{k: v for k, v in asdict(execution).items() if k != 'timestamp'}
            )
            
            self.session.add(record)
            self.session.commit()
            
        except Exception as e:
            print(f"⚠️  Failed to collect test result: {e}")
            self.session.rollback()
    
    def get_changed_files(self, commit_sha: str) -> List[str]:
        """Get files changed in commit using git.
        
        Args:
            commit_sha: Git commit SHA
            
        Returns:
            List of changed file paths
        """
        try:
            result = subprocess.run(
                ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_sha],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            files = result.stdout.strip().split("\n")
            return [f for f in files if f]  # Filter empty strings
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception):
            return []
    
    def calculate_diff_stats(self, commit_sha: str) -> Dict[str, int]:
        """Calculate lines added/deleted in commit.
        
        Args:
            commit_sha: Git commit SHA
            
        Returns:
            Dict with 'lines_added' and 'lines_deleted'
        """
        try:
            result = subprocess.run(
                ["git", "show", "--stat", commit_sha],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            
            # Parse output like: "10 files changed, 150 insertions(+), 50 deletions(-)"
            stats_line = [line for line in result.stdout.split("\n") 
                         if "files changed" in line or "file changed" in line]
            
            if stats_line:
                parts = stats_line[0].split(",")
                lines_added = 0
                lines_deleted = 0
                
                for part in parts:
                    if "insertion" in part:
                        lines_added = int(''.join(filter(str.isdigit, part)))
                    elif "deletion" in part:
                        lines_deleted = int(''.join(filter(str.isdigit, part)))
                
                return {"lines_added": lines_added, "lines_deleted": lines_deleted}
        
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception):
            pass
        
        return {"lines_added": 0, "lines_deleted": 0}
    
    def calculate_complexity_delta(self, file_path: str, commit_sha: str) -> float:
        """Calculate cyclomatic complexity change for a file.
        
        Args:
            file_path: Path to file
            commit_sha: Git commit SHA
            
        Returns:
            Complexity delta (approximate)
        """
        # Simplified: use file size change as proxy for complexity
        # Full implementation would use radon or similar tool
        try:
            if not Path(file_path).exists():
                return 0.0
            
            # Get current file size
            current_size = Path(file_path).stat().st_size
            
            # Get previous file size
            try:
                result = subprocess.run(
                    ["git", "show", f"{commit_sha}~1:{file_path}"],
                    capture_output=True,
                    check=True,
                    timeout=2
                )
                prev_size = len(result.stdout)
                
                # Normalize by 1000 bytes
                delta = (current_size - prev_size) / 1000.0
                return delta
            
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return 0.0
        
        except Exception:
            return 0.0
    
    def get_recent_failure_rate(self, test_name: str, days: int = 30) -> float:
        """Calculate recent failure rate for a test.
        
        Args:
            test_name: Full test name (pytest node ID)
            days: Number of days to look back
            
        Returns:
            Failure rate (0.0 to 1.0)
        """
        try:
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
                    TestTelemetry.passed == False  # noqa: E712
                )
            ).count()
            
            return failures / total
        
        except Exception:
            return 0.0
    
    def get_avg_duration(self, test_name: str, limit: int = 10) -> float:
        """Get average duration for a test from recent history.
        
        Args:
            test_name: Full test name (pytest node ID)
            limit: Number of recent executions to consider
            
        Returns:
            Average duration in milliseconds
        """
        try:
            recent = self.session.query(TestTelemetry.duration_ms).filter(
                TestTelemetry.test_name == test_name
            ).order_by(
                TestTelemetry.created_at.desc()
            ).limit(limit).all()
            
            if not recent:
                return 0.0
            
            return sum(r[0] for r in recent) / len(recent)
        
        except Exception:
            return 0.0
    
    def get_days_since_last_change(self, file_path: str) -> int:
        """Get days since file was last modified.
        
        Args:
            file_path: Path to test file
            
        Returns:
            Number of days since last modification
        """
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ct", file_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=2
            )
            
            if result.stdout.strip():
                last_modified = int(result.stdout.strip())
                now = int(time.time())
                days = (now - last_modified) / 86400  # seconds in a day
                return int(days)
        
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception):
            pass
        
        return 0
    
    def export_training_data(self, output_path: Path, limit: Optional[int] = None) -> None:
        """Export all test telemetry for ML training.
        
        Args:
            output_path: Path to write JSON file
            limit: Optional limit on number of records (for testing)
        """
        try:
            query = self.session.query(TestTelemetry).order_by(
                TestTelemetry.created_at.desc()
            )
            
            if limit:
                query = query.limit(limit)
            
            records = query.all()
            
            data = [
                {
                    "test_name": r.test_name,
                    "test_file": r.test_file,
                    "duration_ms": r.duration_ms,
                    "passed": r.passed,
                    "commit_sha": r.commit_sha,
                    "branch": r.branch,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
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
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(data, indent=2))
            
            print(f"✅ Exported {len(data)} test records to {output_path}")
            return len(data)
        
        except Exception as e:
            print(f"❌ Failed to export training data: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, any]:
        """Get telemetry statistics.
        
        Returns:
            Dict with statistics about collected data
        """
        try:
            total = self.session.query(TestTelemetry).count()
            passed = self.session.query(TestTelemetry).filter(
                TestTelemetry.passed == True  # noqa: E712
            ).count()
            failed = total - passed
            
            unique_tests = self.session.query(
                sql_func.count(sql_func.distinct(TestTelemetry.test_name))
            ).scalar()
            
            avg_duration = self.session.query(
                sql_func.avg(TestTelemetry.duration_ms)
            ).scalar() or 0.0
            
            return {
                "total_executions": total,
                "passed": passed,
                "failed": failed,
                "failure_rate": failed / total if total > 0 else 0.0,
                "unique_tests": unique_tests,
                "avg_duration_ms": float(avg_duration),
            }
        
        except Exception as e:
            print(f"⚠️  Failed to get statistics: {e}")
            return {}
    
    @staticmethod
    def _generate_id(execution: TestExecution) -> str:
        """Generate unique ID for test execution.
        
        Args:
            execution: TestExecution instance
            
        Returns:
            16-character hex string
        """
        key = f"{execution.test_name}:{execution.commit_sha}:{execution.timestamp}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


def collect_from_environment() -> TestExecution:
    """Helper to collect test execution data from environment variables.
    
    This is useful for CI/CD where test context is set via env vars.
    
    Returns:
        TestExecution with data from environment
    """
    collector = TestCollector()
    
    commit_sha = os.getenv("GITHUB_SHA", os.getenv("CI_COMMIT_SHA", "local"))
    branch = os.getenv("GITHUB_REF_NAME", os.getenv("CI_BRANCH", "local"))
    
    changed_files = []
    if commit_sha != "local":
        changed_files = collector.get_changed_files(commit_sha)
    
    diff_stats = collector.calculate_diff_stats(commit_sha) if commit_sha != "local" else {}
    
    return TestExecution(
        test_name="",  # Will be filled by pytest plugin
        test_file="",  # Will be filled by pytest plugin
        duration_ms=0.0,  # Will be filled by pytest plugin
        passed=True,  # Will be filled by pytest plugin
        commit_sha=commit_sha,
        branch=branch,
        changed_files=changed_files,
        lines_added=diff_stats.get("lines_added", 0),
        lines_deleted=diff_stats.get("lines_deleted", 0),
        files_changed=len(changed_files),
    )
