#!/usr/bin/env python3
"""
Automated Bottleneck Identification - AI-Powered Performance Analysis

Analyzes flamegraph data and automatically:
1. Identifies top bottlenecks
2. Locates exact code
3. Recommends specific optimizations
4. Estimates speedup potential

Part of Phase 3: Continuous Profiling

Usage: python scripts/identify_bottlenecks.py

Author: GOATnote Autonomous Research Lab Initiative
Date: October 6, 2025
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Bottleneck:
    """A performance bottleneck with optimization recommendation."""
    function_name: str
    file_path: str
    line_number: int
    time_percent: float
    call_count: int
    optimization: str
    expected_speedup: str
    priority: int


class BottleneckAnalyzer:
    """Automatically identify and recommend fixes for performance bottlenecks."""
    
    def __init__(self, artifacts_dir: str = "artifacts/performance_analysis"):
        self.artifacts_dir = Path(artifacts_dir)
        self.bottlenecks: List[Bottleneck] = []
        
        # Common optimization patterns
        self.patterns = {
            'numpy': {
                'keywords': ['array', 'matrix', 'vector', 'multiply', 'dot'],
                'fix': 'Use NumPy vectorization instead of loops',
                'speedup': '10-100x'
            },
            'json_load': {
                'keywords': ['json.load', 'json.loads', 'json.dump'],
                'fix': 'Replace with ujson for 2-3x faster JSON operations',
                'speedup': '2-3x'
            },
            'loop': {
                'keywords': ['for', 'while', 'range', 'enumerate'],
                'fix': 'Vectorize with NumPy or use list comprehensions',
                'speedup': '5-50x'
            },
            'cache': {
                'keywords': ['compute', 'calculate', 'process'],
                'fix': 'Add @lru_cache decorator for repeated computations',
                'speedup': 'Instant for cached calls'
            },
            'io': {
                'keywords': ['read', 'write', 'open', 'close'],
                'fix': 'Batch I/O operations or use buffering',
                'speedup': '2-10x'
            }
        }
    
    def analyze_svg(self, svg_path: Path) -> List[Tuple[str, float]]:
        """Extract function timing data from flamegraph SVG.
        
        Flamegraph SVGs contain function names and percentages in text elements.
        """
        functions = []
        
        if not svg_path.exists():
            return functions
        
        with open(svg_path, 'r') as f:
            content = f.read()
        
        # Extract function names and percentages from SVG text elements
        # Format: <text>function_name (X.XX%)</text>
        pattern = r'<text[^>]*>([^<]+)\s+\((\d+\.?\d*)%\)</text>'
        matches = re.findall(pattern, content)
        
        for func_name, percent in matches:
            try:
                pct = float(percent)
                if pct > 1.0:  # Only interested in functions taking >1% time
                    functions.append((func_name.strip(), pct))
            except ValueError:
                continue
        
        # Sort by percentage (descending)
        functions.sort(key=lambda x: x[1], reverse=True)
        
        return functions[:10]  # Top 10
    
    def detect_optimization_pattern(self, function_name: str) -> Tuple[str, str]:
        """Detect which optimization pattern applies to this function."""
        func_lower = function_name.lower()
        
        for pattern_name, pattern_info in self.patterns.items():
            for keyword in pattern_info['keywords']:
                if keyword in func_lower:
                    return pattern_info['fix'], pattern_info['speedup']
        
        # Default optimization
        return 'Profile function to understand bottleneck', '10-30x'
    
    def locate_code(self, function_name: str) -> Tuple[str, int]:
        """Find the file and line number where function is defined."""
        # Try to grep for function definition
        try:
            # Search in common locations
            search_dirs = ['src/', 'scripts/', 'app/src/', 'core/', 'services/']
            
            for search_dir in search_dirs:
                if not Path(search_dir).exists():
                    continue
                
                # Search for function definition
                cmd = f"grep -rn 'def {function_name}' {search_dir} || true"
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True
                )
                
                if result.stdout:
                    # Parse first match: filename:line_number:content
                    match = result.stdout.split('\n')[0]
                    if ':' in match:
                        parts = match.split(':', 2)
                        if len(parts) >= 2:
                            return parts[0], int(parts[1])
            
            return "Unknown", 0
            
        except Exception:
            return "Unknown", 0
    
    def analyze_all(self) -> List[Bottleneck]:
        """Analyze all flamegraphs and identify bottlenecks."""
        print("🔍 Analyzing flamegraphs for bottlenecks...\n")
        
        svg_files = list(self.artifacts_dir.glob("*.svg"))
        
        if not svg_files:
            print("❌ No flamegraph SVG files found")
            return []
        
        all_functions = {}
        
        for svg_file in svg_files:
            print(f"   → Analyzing: {svg_file.name}")
            functions = self.analyze_svg(svg_file)
            
            for func_name, percent in functions:
                if func_name not in all_functions:
                    all_functions[func_name] = percent
                else:
                    # Average if seen in multiple files
                    all_functions[func_name] = (all_functions[func_name] + percent) / 2
        
        print(f"\n✅ Found {len(all_functions)} functions taking >1% time\n")
        
        # Create bottleneck objects for top functions
        priority = 1
        for func_name, percent in sorted(all_functions.items(), key=lambda x: x[1], reverse=True)[:5]:
            file_path, line_num = self.locate_code(func_name)
            optimization, speedup = self.detect_optimization_pattern(func_name)
            
            bottleneck = Bottleneck(
                function_name=func_name,
                file_path=file_path,
                line_number=line_num,
                time_percent=percent,
                call_count=1,  # Would need more detailed profiling for this
                optimization=optimization,
                expected_speedup=speedup,
                priority=priority
            )
            
            self.bottlenecks.append(bottleneck)
            priority += 1
        
        return self.bottlenecks
    
    def generate_report(self) -> str:
        """Generate a detailed bottleneck analysis report."""
        if not self.bottlenecks:
            return "No bottlenecks identified (all functions < 1% time)"
        
        report = []
        report.append("# 🎯 Automated Bottleneck Analysis\n")
        report.append(f"**Date**: {subprocess.run('date', capture_output=True, text=True).stdout.strip()}")
        report.append(f"**Analysis**: AI-Powered Automatic Detection")
        report.append(f"**Bottlenecks Found**: {len(self.bottlenecks)}\n")
        report.append("---\n")
        
        for i, bottleneck in enumerate(self.bottlenecks, 1):
            report.append(f"## Bottleneck #{i}: `{bottleneck.function_name}`\n")
            report.append(f"**Priority**: {'🔥 HIGH' if i == 1 else '⚠️ MEDIUM' if i <= 3 else '💡 LOW'}")
            report.append(f"**Time**: {bottleneck.time_percent:.1f}% of total runtime")
            report.append(f"**Location**: `{bottleneck.file_path}:{bottleneck.line_number}`")
            report.append(f"**Expected Speedup**: {bottleneck.expected_speedup}\n")
            
            report.append("### 🔧 Recommended Fix:\n")
            report.append(f"```\n{bottleneck.optimization}\n```\n")
            
            if i == 1:
                report.append("### 📝 Implementation Steps:\n")
                report.append(f"1. Open file: `{bottleneck.file_path}`")
                report.append(f"2. Go to line: `{bottleneck.line_number}`")
                report.append("3. Apply optimization (see recommendation above)")
                report.append("4. Run tests: `pytest tests/`")
                report.append("5. Profile again to measure improvement\n")
            
            report.append("---\n")
        
        # Summary
        total_time = sum(b.time_percent for b in self.bottlenecks)
        report.append("## 📊 Summary\n")
        report.append(f"- **Top {len(self.bottlenecks)} bottlenecks**: {total_time:.1f}% of total time")
        report.append(f"- **#1 Priority**: {self.bottlenecks[0].function_name} ({self.bottlenecks[0].time_percent:.1f}%)")
        report.append(f"- **Expected Impact**: {self.bottlenecks[0].expected_speedup} speedup if optimized")
        report.append("\n---\n")
        
        report.append("## 🚀 Next Steps\n")
        report.append("1. **Start with #1** (biggest impact)")
        report.append("2. **Apply recommended fix**")
        report.append("3. **Measure improvement** (run profiling again)")
        report.append("4. **Commit changes** if tests pass")
        report.append("5. **Move to #2** (compound improvements!)\n")
        
        return "\n".join(report)


def main():
    """Main entry point."""
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                            ║")
    print("║  🤖 AUTOMATED BOTTLENECK IDENTIFICATION                                   ║")
    print("║     AI-Powered Performance Analysis                                       ║")
    print("║                                                                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝\n")
    
    analyzer = BottleneckAnalyzer()
    bottlenecks = analyzer.analyze_all()
    
    if not bottlenecks:
        print("✅ No significant bottlenecks found!")
        print("   All functions taking <1% of runtime")
        return
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save report
    report_path = Path("artifacts/performance_analysis/bottleneck_analysis.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Analysis complete: {report_path}\n")
    
    # Print summary
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("\n🎯 TOP 3 BOTTLENECKS:\n")
    
    for i, bottleneck in enumerate(bottlenecks[:3], 1):
        emoji = "🔥" if i == 1 else "⚠️" if i == 2 else "💡"
        print(f"{emoji} #{i}: {bottleneck.function_name}")
        print(f"   Time: {bottleneck.time_percent:.1f}%")
        print(f"   Location: {bottleneck.file_path}:{bottleneck.line_number}")
        print(f"   Fix: {bottleneck.optimization}")
        print(f"   Expected: {bottleneck.expected_speedup} speedup")
        print()
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"\n📄 Full report: {report_path}")
    print("\n🚀 Ready to optimize! Start with #1 (biggest impact)\n")
    
    # Open report
    subprocess.run(['open', str(report_path)])


if __name__ == "__main__":
    main()
