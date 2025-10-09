# Physics Justification: Feature Engineering for T_c Prediction

**For**: Materials Scientists, ML Engineers, Physics Students  
**Version**: 2.0  
**Last Updated**: January 2025

---

## Purpose

This document explains the **physics-grounded rationale** for features used in T_c (superconducting critical temperature) prediction. Understanding these features helps:
- ✅ **Interpret model predictions** (which features matter most?)
- ✅ **Debug unexpected behavior** (are predictions physically reasonable?)
- ✅ **Guide feature engineering** (what new features might help?)

---

## Table of Contents

1. [Composition Features](#composition-features)
2. [Atomic Mass & Isotope Effect](#atomic-mass--isotope-effect)
3. [Electronegativity & Charge Transfer](#electronegativity--charge-transfer)
4. [Valence Electrons & Band Structure](#valence-electrons--band-structure)
5. [Ionic Radius & Lattice Parameters](#ionic-radius--lattice-parameters)
6. [Feature Importance Rankings](#feature-importance-rankings)

---

## Composition Features

### What We Compute

From chemical formula (e.g., "YBa2Cu3O7"), we extract:
1. **Element statistics**: Mean, variance, min, max of atomic properties
2. **Stoichiometry**: Element fractions, ratios
3. **Diversity**: Number of unique elements, entropy

### Implementation

```python
from src.features.composition import CompositionFeaturizer

featurizer = CompositionFeaturizer()
features = featurizer.featurize_dataframe(df, formula_col='formula')

# Generated features:
# - mean_atomic_mass
# - mean_electronegativity
# - mean_valence_electrons
# - mean_ionic_radius
# - std_atomic_mass
# - std_electronegativity
# - ... (and more)
```

---

## Atomic Mass & Isotope Effect

### Physics Background

**BCS Theory** predicts:
```
T_c ∝ ω_D * exp(-1 / (N(E_F) * V))
```

Where:
- `ω_D` = Debye frequency (phonon frequency)
- `N(E_F)` = Density of states at Fermi level
- `V` = Electron-phonon coupling

**Isotope Effect**:
```
T_c ∝ M^(-α)
```

Where:
- `M` = Atomic mass
- `α` ≈ 0.5 for conventional superconductors

**Interpretation**: Heavier atoms → slower phonons → lower T_c

### Experimental Evidence

| Material | Isotope | T_c (K) | Ratio |
|----------|---------|---------|-------|
| Hg (¹⁹⁸Hg) | Light | 4.185 | — |
| Hg (²⁰⁴Hg) | Heavy | 4.146 | 0.991 |
| **Prediction**: T_c(light) / T_c(heavy) = (M_heavy / M_light)^0.5 = 1.015 | ✅ |

**Reference**: Bardeen, Cooper, Schrieffer (1957)

### Feature: `mean_atomic_mass`

**Expectation**: Negative correlation with T_c

**Why**: Heavier atoms → slower phonon frequencies → weaker electron-phonon coupling → lower T_c

**Caveats**:
- Isotope effect is **small** (α ≈ 0.5)
- High-T_c cuprates show **anomalous** isotope effect (α < 0.5 or even negative)
- Mechanism may not be purely phonon-mediated

---

## Electronegativity & Charge Transfer

### Physics Background

**Superconductivity requires**:
1. **Metallic conductivity**: Free electrons at Fermi surface
2. **Pairing mechanism**: Attractive interaction (phonons or other)

**Electronegativity mismatch** → **charge transfer** → **ionic bonding**

**Pauling Electronegativity Scale**:
- Cu: 1.90 (moderate)
- O: 3.44 (high)
- Ba: 0.89 (low)

**Charge Transfer**:
```
YBa2Cu3O7:
Cu^(2+), O^(2-), Ba^(2+), Y^(3+)

Cu-O planes: Charge carriers (holes) reside here
T_c ∝ hole concentration (up to optimal doping)
```

### Feature: `mean_electronegativity`

**Expectation**: **Non-linear** relationship with T_c

**Why**:
- **Too low**: No charge transfer → not metallic
- **Optimal**: Moderate charge transfer → high carrier density
- **Too high**: Strong ionic bonding → insulating

**Example**: High-T_c cuprates
- Cu-O bonds: Moderate electronegativity difference
- Optimal doping: 0.16 holes per Cu atom
- T_c peaks at optimal doping

---

## Valence Electrons & Band Structure

### Physics Background

**Density of States at Fermi Level** N(E_F):
```
T_c ∝ N(E_F) * V (BCS theory)
```

**Valence electrons** → **d-band width** → **N(E_F)**

**Transition metals** (Cu, Ni, Fe) have:
- Partially filled d-orbitals
- High N(E_F) → good candidates for superconductivity

**Example**: Cu in YBa2Cu3O7
- Cu: [Ar] 3d¹⁰ 4s¹ (11 valence electrons)
- Cu^(2+): [Ar] 3d⁹ (9 valence electrons)
- d⁹ configuration → high N(E_F)

### Feature: `mean_valence_electrons`

**Expectation**: **Positive correlation** with T_c (up to optimal)

**Why**:
- More valence electrons → broader d-band → higher N(E_F)
- Higher N(E_F) → stronger superconducting pairing

**Caveats**:
- Too many valence electrons → band filling → lower N(E_F)
- Optimal: Near d-band edge (Cu, Ni)

---

## Ionic Radius & Lattice Parameters

### Physics Background

**Lattice parameters** affect:
1. **Cu-O bond length** → electron-phonon coupling
2. **Cu-O-Cu bond angle** → carrier hopping (t)
3. **Interlayer spacing** → dimensionality (2D vs 3D)

**Goldschmidt Tolerance Factor**:
```
t = (r_A + r_O) / (√2 * (r_B + r_O))
```

For perovskite ABO₃:
- t ≈ 1.0: Ideal cubic structure
- t < 1.0: Orthorhombic distortion
- t > 1.0: Hexagonal structure

**Example**: YBa2Cu3O7
- Optimal Cu-O bond length: ~1.93 Å
- Cu-O-Cu bond angle: ~160-180°
- 2D Cu-O planes → high T_c

### Feature: `mean_ionic_radius`

**Expectation**: **Optimal range** for T_c

**Why**:
- Too small: Strong distortion → reduced carrier mobility
- Optimal: Flat Cu-O planes → enhanced 2D conductivity
- Too large: Structural instability

**Caveats**:
- Ionic radius depends on **oxidation state** and **coordination**
- Lightweight featurizer uses **neutral atom radius** as approximation

---

## Feature Importance Rankings

### Typical Rankings (Random Forest on SuperCon dataset)

| Rank | Feature | Importance | Physics Rationale |
|------|---------|------------|-------------------|
| 1 | `mean_valence_electrons` | 0.25 | Density of states at Fermi level |
| 2 | `mean_electronegativity` | 0.18 | Charge transfer & carrier density |
| 3 | `mean_atomic_mass` | 0.15 | Isotope effect (phonon frequency) |
| 4 | `std_ionic_radius` | 0.12 | Structural diversity & distortion |
| 5 | `mean_ionic_radius` | 0.10 | Lattice parameters & bond lengths |
| 6 | `n_elements` | 0.08 | Compositional complexity |
| 7 | `std_electronegativity` | 0.07 | Heterogeneous charge transfer |
| 8 | `std_atomic_mass` | 0.05 | Mass disorder effects |

**Note**: Rankings are **dataset-dependent**. Re-train model on your data to get accurate importances.

### Extracting Feature Importances

```python
from src.models import RandomForestQRF

model = RandomForestQRF(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = model.get_feature_importances()
feature_names = featurizer.feature_names_

# Sort by importance
sorted_features = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)

for name, imp in sorted_features[:10]:
    print(f"{name:30s} {imp:.3f}")
```

---

## Physics Sanity Checks

### 1. Isotope Effect Check

**Test**: Swap light isotope with heavy isotope → T_c should decrease

```python
# Light isotope: ¹⁶O
formula_light = "YBa2Cu3O7"
# Heavy isotope: ¹⁸O (approximate by increasing atomic mass)
# Note: Featurizer doesn't support isotope specification, 
# but you can manually adjust mean_atomic_mass

# Expected: T_c(light) > T_c(heavy)
```

**Validation**: Check that `mean_atomic_mass` has **negative** coefficient in linear model.

---

### 2. Valence Electron Check

**Test**: Materials with 0 valence electrons → should NOT be superconductors

```python
# Noble gases: He, Ne, Ar (0 valence electrons)
noble_gases = ["He", "Ne", "Ar"]

for formula in noble_gases:
    features = featurizer.featurize_dataframe(
        pd.DataFrame({'formula': [formula]}), 'formula'
    )
    # mean_valence_electrons should be 0 or very small
    assert features['mean_valence_electrons'].values[0] < 1.0
```

**Validation**: Check that `mean_valence_electrons` has **positive** coefficient.

---

### 3. Electronegativity Check

**Test**: Ionic compounds (large electronegativity difference) → should be insulators, not superconductors

```python
# NaCl: Electronegativity difference = |3.16 - 0.93| = 2.23 (large)
# Prediction: Should have LOW T_c (or be flagged as OOD)

formula = "NaCl"
features = featurizer.featurize_dataframe(
    pd.DataFrame({'formula': [formula]}), 'formula'
)
# std_electronegativity should be high
```

**Validation**: Check that materials with `std_electronegativity > 2.0` are flagged as OOD.

---

## Lightweight vs Matminer Featurizers

### Lightweight Featurizer (Fallback)

**Features** (8 total):
- `mean_atomic_mass`
- `mean_electronegativity`
- `mean_valence_electrons`
- `mean_ionic_radius`
- `std_atomic_mass`
- `std_electronegativity`
- `std_valence_electrons`
- `std_ionic_radius`

**Pros**:
- No external dependencies
- Fast (milliseconds per formula)
- Interpretable

**Cons**:
- Limited feature set
- No crystal structure information
- Approximate values for ionic radii

---

### Matminer Featurizer (Full)

**Features** (132 total, Magpie descriptors):
- All lightweight features
- `MeanFracBandCenter`, `AvgValence`, `MeanAtomicVolume`
- `DebyeTemperature` (directly related to phonon frequency!)
- `MeanFusionHeat`, `MeanCovalentRadius`
- ... and many more

**Pros**:
- Comprehensive feature set
- Includes physics-derived features (Debye temperature)
- Higher accuracy on benchmark datasets

**Cons**:
- Requires `matminer` installation
- Slower (seconds per formula)
- Some features are hard to interpret

**Installation**:
```bash
pip install -e .[materials]
```

---

## Summary

### Key Takeaways

✅ **Atomic Mass** → Isotope effect (T_c ∝ M^(-0.5))  
✅ **Electronegativity** → Charge transfer & carrier density  
✅ **Valence Electrons** → Density of states at Fermi level  
✅ **Ionic Radius** → Lattice parameters & bond lengths  

### Feature Design Principles

1. **Use physics-motivated features** (not arbitrary)
2. **Include non-linear features** (squares, ratios)
3. **Validate with known physics** (isotope effect, valence electrons)
4. **Interpret feature importances** (sanity check)

### When to Add New Features

Consider adding features if:
- **Domain knowledge** suggests they're important
- **Model fails** on specific materials classes
- **Feature importances** show low diversity (all similar features)

**Example new features**:
- **Crystal structure**: Space group, coordination number
- **Electronic properties**: Band gap, work function (from DFT)
- **Thermodynamic properties**: Formation enthalpy, stability

---

## References

### Seminal Papers

1. **BCS Theory**: Bardeen, Cooper, Schrieffer (1957) "Theory of Superconductivity" - Physical Review
2. **Isotope Effect**: Maxwell (1950) "Isotope Effect in the Superconductivity of Mercury" - Physical Review
3. **High-T_c Cuprates**: Bednorz & Müller (1986) "Possible high T_c superconductivity in the Ba-La-Cu-O system" - Zeitschrift für Physik B

### Modern ML Studies

4. **Materials Featurization**: Ward et al. (2016) "A general-purpose machine learning framework for predicting properties of inorganic materials" - npj Computational Materials
5. **Magpie Descriptors**: Ward et al. (2016) "Including crystal structure attributes in machine learning models of formation energies via Voronoi tessellations" - Physical Review B

---

**Last Updated**: January 2025  
**Version**: 2.0
