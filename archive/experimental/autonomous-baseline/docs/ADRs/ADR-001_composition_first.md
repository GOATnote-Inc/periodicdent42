# ADR-001: Composition-First Feature Strategy

**Status**: Accepted  
**Date**: 2024-10-09  
**Deciders**: Research Team, Autonomous Lab Stakeholders

---

## Context

Superconductor Tc prediction can use multiple data modalities:

1. **Composition** (chemical formula, stoichiometry)
2. **Crystal structure** (space group, lattice parameters, atomic positions)
3. **Synthesis conditions** (temperature, pressure, annealing time)
4. **Electronic structure** (DFT band structure, DOS)
5. **Experimental metadata** (measurement method, sample quality)

For **autonomous lab deployment**, we need to choose the primary feature set for the baseline model.

---

## Decision

**We will use composition-only features as the primary input** for the baseline model.

---

## Rationale

### Advantages of Composition-First

1. **Always Available**
   - Chemical formula is known before synthesis
   - No experimental measurement required
   - No DFT calculation required

2. **Low Computational Cost**
   - Feature extraction: <1 second per compound
   - Model training: <5 minutes for 10,000 compounds
   - Inference: <10 ms per prediction

3. **Physics Grounding**
   - Maps to BCS theory (atomic mass → Debye frequency, valence → N(E_F))
   - Interpretable feature importances
   - Sanity checks on correlations (isotope effect, EN spread)

4. **Leakage-Safe Splitting**
   - Formula-level splitting prevents contamination
   - Family-wise awareness (element sets)
   - Near-duplicate detection via feature similarity

5. **Autonomous Lab Fit**
   - Pre-synthesis screening: Prioritize promising candidates
   - Active learning: Query next compound without waiting for structure refinement
   - Cost-effective: No expensive DFT calculations in the loop

### Disadvantages (Acknowledged)

1. **Lower Ceiling**
   - Structure-aware models (CGCNN, MEGNet) achieve 10-20% better RMSE
   - Ignores polymorphism (same composition, different Tc)

2. **Limited to Known Chemistries**
   - Extrapolation to entirely new element combinations is risky
   - OOD detection critical

3. **No Mechanistic Insight**
   - Cannot explain *why* a structure is favorable
   - Requires domain expert review for novel predictions

---

## Alternatives Considered

### Alternative 1: Structure-First

**Pros**: Higher accuracy, captures polymorphism  
**Cons**:
- Structure must be known (DFT relaxation or experimental refinement)
- Adds 10-100x computational cost
- Not available pre-synthesis

**Verdict**: Rejected for baseline; consider for Phase 2 enhancement

---

### Alternative 2: Multi-Modal (Composition + Structure)

**Pros**: Best accuracy, handles polymorphism  
**Cons**:
- Requires structure for inference (not always available)
- Complicates active learning (query structure or composition?)
- Increases model complexity (graph NN + tabular fusion)

**Verdict**: Rejected for baseline; consider for multi-fidelity extension

---

### Alternative 3: Synthesis-Condition-Aware

**Pros**: Captures process-structure-property links  
**Cons**:
- Requires standardized synthesis protocols (not available in literature data)
- Confounds materials properties with synthesis artifacts
- Limited transferability across labs

**Verdict**: Rejected; requires domain-specific dataset

---

## Consequences

### Positive

- ✅ **Fast iteration**: Can screen 1000s of candidates per minute
- ✅ **Interpretable**: Features map to physics (BCS theory)
- ✅ **Robust splitting**: Leakage prevention via formula/family checks
- ✅ **Autonomous-lab ready**: No human-in-loop for feature extraction

### Negative

- ⚠️ **Accuracy ceiling**: Will underperform structure-aware models by 10-20% RMSE
- ⚠️ **Polymorphism blind**: Cannot distinguish Ca₂CuO₃ polymorphs
- ⚠️ **Limited novelty**: Extrapolation to new element combinations requires expert review

### Mitigation Strategies

1. **Ensemble with structure models**: Use composition model for pre-screening, structure model for refinement
2. **Active learning with diversity**: Ensure chemical space coverage despite accuracy limitations
3. **OOD detection**: Flag extrapolations aggressively, query domain expert
4. **Continuous validation**: Monitor performance on lab-generated data, retrain quarterly

---

## Validation Criteria

- ✅ Feature extraction <1s per compound (tested on 10,000 compounds)
- ✅ RMSE within 20% of state-of-art structure models (literature benchmark)
- ✅ Top 5 features align with BCS intuition (SHAP validation)
- ✅ Active learning reduces RMSE by ≥30% in 20 acquisitions (vs random)

---

## Related ADRs

- ADR-002: Uncertainty Calibration Strategy
- ADR-003: Active Learning Strategy

---

## References

- Hamidieh, K. (2018). *A data-driven statistical model for predicting the critical temperature of a superconductor*. Computational Materials Science, 154, 346-354.
- Stanev, V., et al. (2018). *Machine learning modeling of superconducting critical temperature*. npj Computational Materials, 4, 29.
- Ward, L., et al. (2016). *A general-purpose machine learning framework for predicting properties of inorganic materials*. npj Computational Materials, 2, 16028.

---

**Supersedes**: None  
**Superseded by**: None  
**Status**: Active

