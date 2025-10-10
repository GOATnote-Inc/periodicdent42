"""Tests for scripts/repo_audit.py - Repository audit heuristics."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from repo_audit import (
    Finding,
    iter_python_files,
    has_tests_for,
    scan_file,
    build_findings,
    EXCLUDE_DIRS,
)


class TestFinding:
    """Test Finding dataclass."""
    
    def test_finding_creation(self):
        """Test creating Finding instance."""
        finding = Finding(
            path="app/main.py",
            category="Testing",
            issue="Missing tests",
            evidence="No test file",
            recommendation="Add tests",
            effort=2,
            priority="High"
        )
        
        assert finding.path == "app/main.py"
        assert finding.category == "Testing"
        assert finding.issue == "Missing tests"
        assert finding.evidence == "No test file"
        assert finding.recommendation == "Add tests"
        assert finding.effort == 2
        assert finding.priority == "High"
    
    def test_finding_is_dataclass(self):
        """Test that Finding has dataclass features."""
        from dataclasses import asdict
        
        finding = Finding(
            path="test.py",
            category="Code Hygiene",
            issue="TODO",
            evidence="Found TODO",
            recommendation="Fix TODO",
            effort=1,
            priority="Medium"
        )
        
        # Should be convertible to dict
        finding_dict = asdict(finding)
        assert finding_dict["path"] == "test.py"
        assert finding_dict["category"] == "Code Hygiene"


class TestIterPythonFiles:
    """Test iter_python_files function."""
    
    def test_iter_empty_directory(self):
        """Test iterating over empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            files = list(iter_python_files(root))
            
            assert files == []
    
    def test_iter_finds_python_files(self):
        """Test finding Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create Python files
            (root / "test1.py").write_text("# test1")
            (root / "test2.py").write_text("# test2")
            (root / "notpython.txt").write_text("text")
            
            files = sorted(iter_python_files(root))
            
            assert len(files) == 2
            assert files[0].name == "test1.py"
            assert files[1].name == "test2.py"
    
    def test_iter_finds_nested_python_files(self):
        """Test finding Python files in nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create nested structure
            (root / "src").mkdir()
            (root / "src" / "module1.py").write_text("# module1")
            (root / "src" / "subdir").mkdir()
            (root / "src" / "subdir" / "module2.py").write_text("# module2")
            
            files = sorted(iter_python_files(root))
            
            assert len(files) == 2
    
    def test_iter_excludes_git_directory(self):
        """Test that .git directory is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create .git directory with Python file
            (root / ".git").mkdir()
            (root / ".git" / "hook.py").write_text("# git hook")
            (root / "main.py").write_text("# main")
            
            files = list(iter_python_files(root))
            
            assert len(files) == 1
            assert files[0].name == "main.py"
    
    def test_iter_excludes_node_modules(self):
        """Test that node_modules directory is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            (root / "node_modules").mkdir()
            (root / "node_modules" / "script.py").write_text("# node script")
            (root / "app.py").write_text("# app")
            
            files = list(iter_python_files(root))
            
            assert len(files) == 1
            assert files[0].name == "app.py"
    
    def test_iter_excludes_pycache(self):
        """Test that __pycache__ directory is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "cached.py").write_text("# cached")
            (root / "source.py").write_text("# source")
            
            files = list(iter_python_files(root))
            
            assert len(files) == 1
            assert files[0].name == "source.py"
    
    def test_iter_excludes_all_excluded_dirs(self):
        """Test that all EXCLUDE_DIRS are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create all excluded directories
            for exclude_dir in EXCLUDE_DIRS:
                (root / exclude_dir).mkdir()
                (root / exclude_dir / "file.py").write_text(f"# in {exclude_dir}")
            
            (root / "valid.py").write_text("# valid")
            
            files = list(iter_python_files(root))
            
            assert len(files) == 1
            assert files[0].name == "valid.py"


class TestHasTestsFor:
    """Test has_tests_for function."""
    
    def test_no_tests_directory(self):
        """Test when tests directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_file = root / "module.py"
            source_file.write_text("# module")
            tests_root = root / "tests"
            
            result = has_tests_for(source_file, tests_root)
            
            assert result == False
    
    def test_has_test_with_matching_name(self):
        """Test finding test file with matching module name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_file = root / "calculator.py"
            source_file.write_text("def add(a, b): return a + b")
            
            tests_root = root / "tests"
            tests_root.mkdir()
            test_file = tests_root / "test_calculator.py"
            test_file.write_text("from calculator import add")
            
            result = has_tests_for(source_file, tests_root)
            
            assert result == True
    
    def test_has_test_nested_directory(self):
        """Test finding test in nested test directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_file = root / "utils.py"
            source_file.write_text("def helper(): pass")
            
            tests_root = root / "tests"
            (tests_root / "unit").mkdir(parents=True)
            test_file = tests_root / "unit" / "test_helpers.py"
            test_file.write_text("import utils")
            
            result = has_tests_for(source_file, tests_root)
            
            assert result == True
    
    def test_no_matching_test(self):
        """Test when no test file references the module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_file = root / "forgotten.py"
            source_file.write_text("def func(): pass")
            
            tests_root = root / "tests"
            tests_root.mkdir()
            test_file = tests_root / "test_other.py"
            test_file.write_text("import something_else")
            
            result = has_tests_for(source_file, tests_root)
            
            assert result == False
    
    def test_multiple_test_files_one_matches(self):
        """Test with multiple test files, one matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_file = root / "target.py"
            source_file.write_text("def func(): pass")
            
            tests_root = root / "tests"
            tests_root.mkdir()
            (tests_root / "test_other.py").write_text("import other")
            (tests_root / "test_target.py").write_text("from target import func")
            
            result = has_tests_for(source_file, tests_root)
            
            assert result == True
    
    def test_word_boundary_matching(self):
        """Test that module name matching respects word boundaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_file = root / "config.py"
            source_file.write_text("CONFIG = {}")
            
            tests_root = root / "tests"
            tests_root.mkdir()
            # Should NOT match "reconfig" or "_config"
            test_file = tests_root / "test_app.py"
            test_file.write_text("from reconfig import settings")
            
            result = has_tests_for(source_file, tests_root)
            
            assert result == False


class TestScanFile:
    """Test scan_file function."""
    
    def test_scan_empty_file(self):
        """Test scanning empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "empty.py"
            file_path.write_text("")
            
            metrics = scan_file(file_path)
            
            assert metrics["todo_count"] == 0
            assert metrics["long_functions"] == 0
    
    def test_scan_file_with_todos(self):
        """Test detecting TODO comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "todos.py"
            file_path.write_text("""
def func1():
    # TODO: implement this
    pass

def func2():
    # FIXME: broken logic
    pass
""")
            
            metrics = scan_file(file_path)
            
            assert metrics["todo_count"] == 2
    
    def test_scan_file_todo_case_insensitive(self):
        """Test that TODO matching is case insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "mixed.py"
            file_path.write_text("""
# Todo: lowercase
# TODO: uppercase
# todo: all lowercase
""")
            
            metrics = scan_file(file_path)
            
            assert metrics["todo_count"] == 3
    
    def test_scan_file_short_function(self):
        """Test detecting short functions (not counted as long)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "short.py"
            file_path.write_text("""
def short_func():
    x = 1
    y = 2
    return x + y
""")
            
            metrics = scan_file(file_path)
            
            assert metrics["long_functions"] == 0
    
    def test_scan_file_long_function(self):
        """Test detecting functions exceeding 100 lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "long.py"
            # Create function with 101 lines
            lines = ["def long_func():"] + ["    pass"] * 101
            file_path.write_text("\n".join(lines))
            
            metrics = scan_file(file_path)
            
            assert metrics["long_functions"] == 1
    
    def test_scan_file_multiple_long_functions(self):
        """Test detecting multiple long functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "multiple.py"
            func1_lines = ["def func1():"] + ["    pass"] * 101
            func2_lines = ["def func2():"] + ["    pass"] * 101
            file_path.write_text("\n".join(func1_lines + func2_lines))
            
            metrics = scan_file(file_path)
            
            assert metrics["long_functions"] == 2
    
    def test_scan_file_class_ends_function(self):
        """Test that class declaration ends function counting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "with_class.py"
            file_path.write_text("""
def short_func():
    pass

class MyClass:
    pass
""")
            
            metrics = scan_file(file_path)
            
            assert metrics["long_functions"] == 0


class TestBuildFindings:
    """Test build_findings function."""
    
    def test_build_findings_empty_repo(self):
        """Test building findings for empty repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            findings = build_findings(root)
            
            assert findings == []
    
    def test_build_findings_missing_tests_high_priority(self):
        """Test that API files without tests get high priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create API file without tests
            api_dir = root / "services" / "api"
            api_dir.mkdir(parents=True)
            (api_dir / "endpoints.py").write_text("def get_users(): pass")
            
            findings = build_findings(root)
            
            assert len(findings) == 1
            assert findings[0].category == "Testing"
            assert findings[0].priority == "High"
            assert "api" in findings[0].path.lower()
    
    def test_build_findings_missing_tests_medium_priority(self):
        """Test that non-API files without tests get medium priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create service file without tests
            services_dir = root / "services"
            services_dir.mkdir()
            (services_dir / "helper.py").write_text("def util(): pass")
            
            findings = build_findings(root)
            
            assert len(findings) == 1
            assert findings[0].category == "Testing"
            assert findings[0].priority == "Medium"
    
    def test_build_findings_with_tests(self):
        """Test that files with tests are not flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create service file
            services_dir = root / "services"
            services_dir.mkdir()
            (services_dir / "calculator.py").write_text("def add(a, b): return a + b")
            
            # Create test file
            tests_dir = root / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_calculator.py").write_text("from calculator import add")
            
            findings = build_findings(root)
            
            # Should not flag as missing tests
            testing_findings = [f for f in findings if f.category == "Testing"]
            assert len(testing_findings) == 0
    
    def test_build_findings_todo_comments(self):
        """Test detecting TODO comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            services_dir = root / "services"
            services_dir.mkdir()
            (services_dir / "work.py").write_text("""
def func():
    # TODO: implement
    pass
""")
            
            findings = build_findings(root)
            
            hygiene_findings = [f for f in findings if f.category == "Code Hygiene"]
            assert len(hygiene_findings) == 1
            assert "TODO" in hygiene_findings[0].issue
            assert hygiene_findings[0].effort == 1
    
    def test_build_findings_long_functions(self):
        """Test detecting long functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            services_dir = root / "services"
            services_dir.mkdir()
            long_func = ["def long():"] + ["    pass"] * 101
            (services_dir / "big.py").write_text("\n".join(long_func))
            
            findings = build_findings(root)
            
            maintainability_findings = [f for f in findings if f.category == "Maintainability"]
            assert len(maintainability_findings) == 1
            assert "100 lines" in maintainability_findings[0].issue
            assert maintainability_findings[0].effort == 3
    
    def test_build_findings_multiple_issues(self):
        """Test file with multiple issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            services_dir = root / "services"
            services_dir.mkdir()
            
            # File with TODO and long function
            long_func_with_todo = (
                ["def long():"] +
                ["    # TODO: refactor"] +
                ["    pass"] * 100
            )
            (services_dir / "messy.py").write_text("\n".join(long_func_with_todo))
            
            findings = build_findings(root)
            
            # Should have both Testing (no tests), Code Hygiene (TODO), and Maintainability (long)
            assert len(findings) >= 2
            categories = {f.category for f in findings}
            assert "Code Hygiene" in categories
            assert "Maintainability" in categories
    
    def test_build_findings_only_checks_relevant_dirs(self):
        """Test that only services/apps/app directories are checked for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create file in scripts (should not be checked for tests)
            scripts_dir = root / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "tool.py").write_text("def main(): pass")
            
            findings = build_findings(root)
            
            testing_findings = [f for f in findings if f.category == "Testing"]
            # Scripts are not checked for missing tests
            assert len(testing_findings) == 0

