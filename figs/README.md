# Evidence Figures & Visualizations

**Generated**: 2025-10-06  
**Audit**: Evidence Audit for Periodic Labs  
**Source**: Automated analysis + CI artifacts

---

## Available Figures

### Performance Analysis (Flamegraphs)

**1. validate_rl_system_20251006_192536.svg**
- **Script**: `scripts/validate_rl_system.py`
- **Runtime**: 0.204s
- **Analysis**: No functions >1% of total time (well-optimized)
- **Interactive**: Open in browser to zoom/explore

**2. validate_stochastic_20251006_192536.svg**
- **Script**: `scripts/validate_stochastic.py`
- **Runtime**: 0.204s (note: suspiciously identical, may be metadata artifact)
- **Analysis**: No significant bottlenecks detected
- **Interactive**: Open in browser to zoom/explore

---

## How to Use Flamegraphs

1. **Open SVG in browser** (Chrome, Firefox, Safari)
2. **Look for WIDE bars** - These represent functions taking lots of time
3. **Hover over bars** - See function names and time percentages
4. **Click bars** - Zoom into specific call stacks
5. **Click "Reset Zoom"** - Return to full view

### What to Optimize
- **Widest bars at any level** - Biggest bottlenecks
- **Unexpectedly slow functions** - Logic that should be faster
- **Repeated patterns** - Caching opportunities

### Performance Targets
- **>1% of runtime** - Worth investigating
- **>5% of runtime** - Definitely optimize
- **>10% of runtime** - High-priority optimization

---

## Missing Figures (To Be Generated)

### C1: Hermetic Builds
- [ ] `build_times.png` - Distribution of build times across CI runs
- [ ] `platform_matrix.png` - Cross-platform hash comparison table
- [ ] `cache_hit_rates.png` - Nix cache performance over time

**Data Needed**: 10+ CI runs with Nix builds

### C2: ML Test Selection
- [ ] `precision_recall_curve.png` - PR curve with operating point
- [ ] `utility_curve.png` - Expected utility vs threshold
- [ ] `feature_importance.png` - RandomForest feature importance

**Data Needed**: Retrain model on real data (50+ test runs)

### C3: Chaos Engineering
- [ ] `chaos_pass_rates.png` - Pass rates vs chaos injection rate
- [ ] `failure_taxonomy.png` - Failure types vs resilience patterns (heatmap)
- [ ] `runtime_overhead.png` - Chaos injection performance impact

**Data Needed**: Multiple chaos test runs at different rates

### C4: Continuous Profiling
- [ ] `perf_trend.png` - Performance over time with regression detection
- [ ] `bottleneck_comparison.png` - Before/after optimization comparison
- [ ] `manual_vs_ai_timing.png` - Manual analysis time vs AI (N=5)

**Data Needed**: 20+ CI runs with profiling enabled

---

## Generating Missing Figures

### Prerequisites
```bash
pip install matplotlib seaborn pandas numpy scipy
```

### C2: ML Precision-Recall Curve
```bash
python scripts/generate_ml_figures.py \
  --model test_selector.pkl \
  --data training_data.json \
  --output figs/
```

### C3: Chaos Pass Rates
```bash
python scripts/generate_chaos_figures.py \
  --test-dir tests/chaos/ \
  --rates 0.0,0.05,0.10,0.15,0.20 \
  --output figs/
```

### C4: Performance Trends
```bash
python scripts/generate_perf_figures.py \
  --artifacts artifacts/performance_analysis/ \
  --output figs/
```

---

## Figure Specifications

All figures should follow these standards:

### Resolution
- **Vector**: SVG preferred (scalable, small file size)
- **Raster**: PNG at 300 DPI minimum
- **Size**: 1200x800 pixels minimum

### Style
- **Font**: Sans-serif (Arial, Helvetica)
- **Font Size**: 12pt minimum for body, 16pt for titles
- **Colors**: Colorblind-safe palette (viridis, plasma)
- **Grid**: Light gray, dashed
- **Legend**: Upper right or below plot

### Annotations
- **Title**: Descriptive (e.g., "ML Test Selection: Precision-Recall Curve")
- **Axes**: Labeled with units
- **Data Points**: N= sample size noted
- **Confidence Intervals**: Shaded regions (95% CI)
- **Source**: Footer with generation date and commit

---

## Current Status

**Available**: 2 flamegraphs (performance analysis)  
**Missing**: 10 figures (need data collection)  
**Action**: Collect 2 weeks production data → regenerate figures

---

© 2025 GOATnote Autonomous Research Lab Initiative  
Figures Index: 2025-10-06
