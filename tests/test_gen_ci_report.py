"""Tests for scripts/gen_ci_report.py - CI report generation."""

import json
import tempfile
from pathlib import Path
import sys

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_import_gen_ci_report():
    """Test that gen_ci_report module can be imported."""
    try:
        import gen_ci_report
        assert hasattr(gen_ci_report, "__file__")
    except ImportError as e:
        pytest.skip(f"gen_ci_report not available: {e}")


def test_generate_summary_stats():
    """Test summary statistics generation."""
    from gen_ci_report import generate_summary_stats
    
    selected_tests = [
        {"name": "test_a", "eig_score": 0.9, "duration_ms": 1000},
        {"name": "test_b", "eig_score": 0.8, "duration_ms": 2000},
    ]
    
    all_tests = [
        {"name": "test_a", "eig_score": 0.9, "duration_ms": 1000},
        {"name": "test_b", "eig_score": 0.8, "duration_ms": 2000},
        {"name": "test_c", "eig_score": 0.5, "duration_ms": 1000},
    ]
    
    stats = generate_summary_stats(selected_tests, all_tests)
    
    assert "num_selected" in stats
    assert "num_total" in stats
    assert "coverage_pct" in stats
    assert "time_saved_sec" in stats
    
    assert stats["num_selected"] == 2
    assert stats["num_total"] == 3
    assert stats["coverage_pct"] == pytest.approx(66.67, abs=0.1)


def test_compute_cost_savings():
    """Test cost savings computation."""
    from gen_ci_report import compute_cost_savings
    
    full_suite_duration_sec = 300  # 5 minutes
    selected_duration_sec = 120     # 2 minutes
    runner_usd_per_hour = 0.60
    
    savings = compute_cost_savings(
        full_suite_duration_sec,
        selected_duration_sec,
        runner_usd_per_hour
    )
    
    assert "time_saved_sec" in savings
    assert "time_reduction_pct" in savings
    assert "cost_saved_usd" in savings
    
    assert savings["time_saved_sec"] == 180  # 3 minutes
    assert savings["time_reduction_pct"] == 60.0
    
    # Cost calculation: (180s / 3600s) * $0.60 = $0.03
    assert savings["cost_saved_usd"] == pytest.approx(0.03, abs=0.001)


def test_format_markdown_report():
    """Test Markdown report formatting."""
    from gen_ci_report import format_markdown_report
    
    metrics = {
        "num_selected": 50,
        "num_total": 100,
        "coverage_pct": 50.0,
        "time_saved_sec": 180,
        "cost_saved_usd": 0.03,
        "total_eig": 25.5,
        "avg_eig": 0.51,
    }
    
    report = format_markdown_report(metrics)
    
    # Should be valid Markdown
    assert "##" in report or "#" in report
    assert "50/100" in report or "50" in report
    assert "50%" in report or "50.0" in report


def test_generate_json_metrics():
    """Test JSON metrics generation."""
    from gen_ci_report import generate_json_metrics
    
    data = {
        "selected": [
            {"name": "test_a", "eig_score": 0.9},
            {"name": "test_b", "eig_score": 0.8},
        ],
        "all_tests": [
            {"name": "test_a", "eig_score": 0.9},
            {"name": "test_b", "eig_score": 0.8},
            {"name": "test_c", "eig_score": 0.5},
        ],
        "full_suite_duration_sec": 300,
        "selected_duration_sec": 120,
    }
    
    metrics = generate_json_metrics(data)
    
    # Should be valid JSON-serializable
    json_str = json.dumps(metrics)
    loaded = json.loads(json_str)
    
    assert loaded == metrics
    
    # Should have required fields
    assert "timestamp" in metrics
    assert "num_selected" in metrics
    assert "cost_reduction_pct" in metrics


def test_save_report_files():
    """Test saving report to multiple formats."""
    from gen_ci_report import save_reports
    
    metrics = {
        "num_selected": 50,
        "num_total": 100,
        "coverage_pct": 50.0,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        save_reports(metrics, output_dir)
        
        # Check files were created
        assert (output_dir / "ci_report.md").exists()
        assert (output_dir / "ci_metrics.json").exists()
        
        # Verify content
        md_content = (output_dir / "ci_report.md").read_text()
        assert "50" in md_content
        
        json_content = json.loads((output_dir / "ci_metrics.json").read_text())
        assert json_content["num_selected"] == 50


def test_compute_information_theory_metrics():
    """Test information-theoretic metrics computation."""
    from gen_ci_report import compute_information_metrics
    
    selected_tests = [
        {"name": "test_a", "eig_score": 0.9, "failure_prob": 0.5},
        {"name": "test_b", "eig_score": 0.7, "failure_prob": 0.3},
    ]
    
    metrics = compute_information_metrics(selected_tests)
    
    assert "total_eig" in metrics
    assert "avg_eig" in metrics
    assert "max_eig" in metrics
    assert "entropy" in metrics
    
    assert metrics["total_eig"] == pytest.approx(1.6, abs=0.1)
    assert metrics["avg_eig"] == pytest.approx(0.8, abs=0.1)


def test_generate_test_table():
    """Test test table generation for report."""
    from gen_ci_report import generate_test_table
    
    tests = [
        {"name": "test_a", "eig_score": 0.9, "duration_ms": 1000, "rank": 1},
        {"name": "test_b", "eig_score": 0.8, "duration_ms": 2000, "rank": 2},
    ]
    
    table = generate_test_table(tests, max_rows=10)
    
    # Should be Markdown table
    assert "|" in table
    assert "test_a" in table
    assert "test_b" in table
    assert "0.9" in table or "0.90" in table


def test_add_timestamp_to_report():
    """Test that reports include timestamps."""
    from gen_ci_report import generate_json_metrics
    from datetime import datetime
    
    metrics = generate_json_metrics({
        "selected": [],
        "all_tests": [],
        "full_suite_duration_sec": 0,
        "selected_duration_sec": 0,
    })
    
    assert "timestamp" in metrics
    
    # Should be ISO format
    timestamp = metrics["timestamp"]
    assert "T" in timestamp
    
    # Should be recent (within last minute)
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    now = datetime.now(dt.tzinfo)
    diff = (now - dt).total_seconds()
    assert diff < 60  # Generated within last minute


def test_compute_ledger_metrics():
    """Test ledger emission metrics computation."""
    from gen_ci_report import compute_ledger_metrics
    
    data = {
        "num_runs": 100,
        "total_duration_hours": 5.0,
        "total_cost_usd": 3.00,
        "co2_kg": 0.5,
    }
    
    ledger = compute_ledger_metrics(data)
    
    assert "runs" in ledger
    assert "duration_hours" in ledger
    assert "cost_usd" in ledger
    assert "co2_kg" in ledger
    
    # Should include per-run averages
    assert "avg_duration_min" in ledger
    assert "avg_cost_usd" in ledger


def test_format_duration():
    """Test duration formatting utility."""
    from gen_ci_report import format_duration
    
    assert format_duration(30) == "30s"
    assert format_duration(90) == "1m 30s"
    assert format_duration(3665) == "1h 1m 5s"


def test_report_reproducibility():
    """Test that same inputs produce same report."""
    from gen_ci_report import generate_summary_stats
    
    selected = [{"name": "test_a", "eig_score": 0.9, "duration_ms": 1000}]
    all_tests = [{"name": "test_a", "eig_score": 0.9, "duration_ms": 1000}]
    
    stats1 = generate_summary_stats(selected, all_tests)
    stats2 = generate_summary_stats(selected, all_tests)
    
    # Timestamps may differ, remove them
    stats1_no_ts = {k: v for k, v in stats1.items() if k != "timestamp"}
    stats2_no_ts = {k: v for k, v in stats2.items() if k != "timestamp"}
    
    assert stats1_no_ts == stats2_no_ts


def test_handle_empty_selection():
    """Test report generation with no tests selected."""
    from gen_ci_report import generate_summary_stats
    
    stats = generate_summary_stats([], [])
    
    assert stats["num_selected"] == 0
    assert stats["num_total"] == 0
    assert stats["coverage_pct"] == 0.0


def test_validate_metrics_schema():
    """Test that metrics conform to expected schema."""
    from gen_ci_report import generate_json_metrics
    
    metrics = generate_json_metrics({
        "selected": [],
        "all_tests": [],
        "full_suite_duration_sec": 0,
        "selected_duration_sec": 0,
    })
    
    # Required fields
    required = [
        "timestamp", "num_selected", "num_total",
        "coverage_pct", "time_reduction_pct", "cost_reduction_pct"
    ]
    
    for field in required:
        assert field in metrics, f"Missing required field: {field}"

