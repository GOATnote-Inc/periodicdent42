# Performance Analysis Report

**Date**: $(date)
**Run ID**: $RUN_ID
**Analysis**: Automated

---

## 📊 Bottlenecks Identified

### validate

- **Total Duration**: 0.20397067070007324s
- **Flamegraph**: artifacts/performance_analysis/validate_rl_system_20251006_192536.svg


---

## 🔥 How to Use Flamegraphs

1. Open the SVG files in your browser
2. Look for WIDE bars (= taking lots of time)
3. Hover to see function names and percentages
4. Click to zoom into specific functions

**What to optimize**:
- Widest bars at any level (biggest bottlenecks)
- Unexpectedly slow functions
- Repeated patterns (caching opportunities)

