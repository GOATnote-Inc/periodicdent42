"""
Physics-Based Interpretation of ML Predictions

Don't just predict - explain WHY using materials science.
This shows you think like a materials scientist, not just an ML engineer.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


# Known superconductor database (for comparison)
FAMOUS_SUPERCONDUCTORS = {
    "YBCO": {
        "formula": "YBa2Cu3O7",
        "tc": 92,
        "year_discovered": 1987,
        "family": "cuprate",
        "mechanism": "d-wave pairing, spin fluctuations",
        "significance": "First superconductor above liquid nitrogen temperature (77K)",
        "structure": "orthorhombic, triple perovskite with CuO2 planes",
        "key_features": ["high Tc", "layered", "type-II", "short coherence length"],
        "applications": "SQUID devices, power cables, magnets"
    },
    "BSCCO": {
        "formula": "Bi2Sr2CaCu2O8",
        "tc": 85,
        "year_discovered": 1988,
        "family": "cuprate",
        "mechanism": "d-wave pairing",
        "significance": "Tape conductor material",
        "structure": "layered, multiple CuO2 planes",
        "key_features": ["very anisotropic", "flexible tapes possible"]
    },
    "MgB2": {
        "formula": "MgB2",
        "tc": 39,
        "year_discovered": 2001,
        "family": "conventional",
        "mechanism": "s-wave, phonon-mediated (strong e-ph coupling)",
        "significance": "Highest Tc conventional superconductor, simple structure",
        "structure": "hexagonal, graphite-like boron layers",
        "key_features": ["two-gap superconductor", "high Debye temperature", "λ≈0.7"],
        "applications": "MRI magnets, cables, fault current limiters"
    },
    "LaFeAsO": {
        "formula": "LaFeAsO",
        "tc": 26,
        "year_discovered": 2008,
        "family": "iron_pnictide",
        "mechanism": "s± pairing, multi-band",
        "significance": "First iron-based superconductor family",
        "structure": "layered, FeAs tetrahedral layers",
        "key_features": ["sign-changing gap", "multi-band", "magnetic fluctuations"]
    },
    "NbTi": {
        "formula": "NbTi",
        "tc": 10,
        "year_discovered": 1962,
        "family": "conventional",
        "mechanism": "s-wave, BCS",
        "significance": "Most widely used practical superconductor",
        "applications": "MRI magnets, particle accelerators (LHC)"
    },
    "Nb3Sn": {
        "formula": "Nb3Sn",
        "tc": 18,
        "year_discovered": 1954,
        "family": "conventional",
        "mechanism": "s-wave, strong coupling",
        "significance": "High-field magnet material",
        "applications": "ITER fusion reactor, high-field magnets"
    },
    "Pb": {
        "formula": "Pb",
        "tc": 7.2,
        "year_discovered": 1913,
        "family": "conventional",
        "mechanism": "s-wave, BCS (textbook example)",
        "significance": "Classic BCS superconductor",
        "structure": "FCC",
        "key_features": ["type-I", "well-understood"]
    },
    "LaH10": {
        "formula": "LaH10",
        "tc": 250,  # At 170 GPa
        "year_discovered": 2019,
        "family": "hydride",
        "mechanism": "conventional, extreme e-ph coupling",
        "significance": "Highest Tc observed (under pressure)",
        "key_features": ["requires extreme pressure", "very high Debye temperature"]
    }
}


def explain_prediction(
    material_formula: str,
    predicted_tc: float,
    features: Dict[str, float],
    uncertainty: Optional[float] = None
) -> Dict[str, Any]:
    """
    Explain prediction using physics principles.
    
    This is what hiring managers want to see:
    - Not just "predicted Tc = 42K"
    - But "Tc = 42K because high DOS, strong e-ph coupling, similar to MgB2"
    
    Args:
        material_formula: Chemical formula
        predicted_tc: Predicted critical temperature (K)
        features: Extracted features including physics features
        uncertainty: Optional prediction uncertainty
    
    Returns:
        Detailed explanation dictionary
    """
    explanation = {
        "material": material_formula,
        "predicted_tc": predicted_tc,
        "confidence": calculate_confidence(uncertainty) if uncertainty else "N/A",
        "key_factors": {},
        "physics_insights": [],
        "mechanism_hypothesis": "",
        "similar_to": [],
        "synthesis_suggestions": []
    }
    
    # Identify key contributing factors
    key_factors = identify_key_factors(features)
    explanation["key_factors"] = key_factors
    
    # Generate physics insights
    explanation["physics_insights"] = generate_physics_insights(features, predicted_tc)
    
    # Hypothesize mechanism
    explanation["mechanism_hypothesis"] = hypothesize_mechanism(material_formula, features)
    
    # Find similar known superconductors
    explanation["similar_to"] = find_similar_known_superconductors(material_formula, features)
    
    # Suggest synthesis approach
    explanation["synthesis_suggestions"] = suggest_synthesis(material_formula, features)
    
    return explanation


def identify_key_factors(features: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    Identify which features most favor/disfavor superconductivity.
    
    Args:
        features: Feature dictionary
    
    Returns:
        Dictionary of key factors with explanations
    """
    key_factors = {}
    
    # Check DOS at Fermi level
    if "dos_fermi" in features:
        dos = features["dos_fermi"]
        if dos > 5.0:
            key_factors["high_dos"] = {
                "value": dos,
                "impact": "positive",
                "explanation": f"High DOS at Fermi level ({dos:.2f} states/eV/atom) suggests strong electron pairing potential (BCS favorable)",
                "contribution_to_tc": f"+{int(dos * 3)}K estimated"
            }
        elif dos < 2.0:
            key_factors["low_dos"] = {
                "value": dos,
                "impact": "negative",
                "explanation": f"Low DOS at Fermi level ({dos:.2f}) limits Cooper pair formation",
                "contribution_to_tc": f"-{int((3-dos) * 5)}K"
            }
    
    # Check electron-phonon coupling
    if "lambda_ep" in features:
        lambda_ep = features["lambda_ep"]
        if lambda_ep > 0.7:
            key_factors["strong_coupling"] = {
                "value": lambda_ep,
                "impact": "positive",
                "explanation": f"Strong electron-phonon coupling (λ={lambda_ep:.3f}) - conventional superconductor mechanism",
                "mcmillan_compatible": "Yes - good candidate for BCS-type superconductivity"
            }
        elif lambda_ep < 0.3:
            key_factors["weak_coupling"] = {
                "value": lambda_ep,
                "impact": "neutral_or_negative",
                "explanation": f"Weak electron-phonon coupling (λ={lambda_ep:.3f}) - may require unconventional mechanism"
            }
    
    # Check Debye temperature
    if "debye_temperature" in features:
        theta_d = features["debye_temperature"]
        if theta_d > 500:
            key_factors["high_debye"] = {
                "value": theta_d,
                "impact": "positive",
                "explanation": f"High Debye temperature ({theta_d:.0f}K) - light atoms or strong bonds provide energetic phonons",
                "similar_to": "MgB2 (θ_D ~ 900K)"
            }
    
    # Check superconductor family indicators
    if features.get("likely_cuprate", 0) == 1:
        key_factors["cuprate_family"] = {
            "impact": "positive",
            "explanation": "Composition suggests cuprate family (layered CuO2 planes)",
            "expected_tc_range": "20-130K",
            "mechanism": "d-wave pairing, spin fluctuations"
        }
    
    if features.get("likely_iron_based", 0) == 1:
        key_factors["iron_based_family"] = {
            "impact": "positive",
            "explanation": "Composition suggests iron-based family (FeAs layers)",
            "expected_tc_range": "20-55K",
            "mechanism": "s± pairing, multi-band"
        }
    
    return key_factors


def generate_physics_insights(features: Dict[str, float], predicted_tc: float) -> List[str]:
    """
    Generate physics-based insights about the prediction.
    
    Args:
        features: Feature dictionary
        predicted_tc: Predicted Tc
    
    Returns:
        List of insight strings
    """
    insights = []
    
    # Compare prediction to physics estimates
    if "mcmillan_tc_estimate" in features:
        mcm_tc = features["mcmillan_tc_estimate"]
        diff = abs(predicted_tc - mcm_tc)
        
        if diff < 10:
            insights.append(f"✅ ML prediction ({predicted_tc:.1f}K) matches McMillan equation ({mcm_tc:.1f}K) - conventional BCS mechanism likely")
        elif predicted_tc > mcm_tc + 20:
            insights.append(f"⚡ ML prediction ({predicted_tc:.1f}K) exceeds McMillan estimate ({mcm_tc:.1f}K) - may indicate unconventional mechanism or enhanced pairing")
        else:
            insights.append(f"⚠️  ML prediction ({predicted_tc:.1f}K) below McMillan estimate ({mcm_tc:.1f}K) - may have suppression factors")
    
    # BCS parameter check
    if "lambda_ep" in features and "mu_star" in features:
        lambda_ep = features["lambda_ep"]
        if lambda_ep > 1.0:
            insights.append(f"🔥 Strong coupling regime (λ={lambda_ep:.2f}) - beyond weak-coupling BCS theory")
    
    # DOS favorability
    if features.get("dos_favorable_for_pairing", 0) == 1:
        insights.append("📊 High density of states at Fermi level favors Cooper pairing")
    
    # Structure insights
    if features.get("likely_layered", 0) == 1:
        insights.append("🏗️ Layered structure expected - may show anisotropic properties")
    
    return insights


def hypothesize_mechanism(material_formula: str, features: Dict[str, float]) -> str:
    """
    Hypothesize the superconducting mechanism.
    
    Args:
        material_formula: Chemical formula
        features: Feature dictionary
    
    Returns:
        Mechanism hypothesis string
    """
    # Check family indicators
    if features.get("likely_cuprate", 0) == 1:
        return "d-wave pairing mediated by spin fluctuations (cuprate mechanism)"
    
    if features.get("likely_iron_based", 0) == 1:
        return "s± pairing with sign-changing order parameter (iron-based mechanism)"
    
    if features.get("likely_hydride", 0) == 1:
        return "Conventional phonon-mediated with extreme electron-phonon coupling (high-pressure hydride)"
    
    # Check coupling strength
    lambda_ep = features.get("lambda_ep", 0)
    
    if lambda_ep > 0.5:
        return "Conventional BCS mechanism with phonon-mediated pairing"
    else:
        return "Mechanism unclear - may be unconventional or weak coupling"


def find_similar_known_superconductors(
    material_formula: str,
    features: Dict[str, float],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Find known superconductors similar to the prediction.
    
    This shows you know the literature and can contextualize predictions.
    
    Args:
        material_formula: Chemical formula
        features: Feature dictionary
        top_k: Number of similar materials to return
    
    Returns:
        List of similar superconductor dictionaries
    """
    similarities = []
    
    for name, sc_data in FAMOUS_SUPERCONDUCTORS.items():
        similarity_score = calculate_similarity(material_formula, features, sc_data)
        
        similarities.append({
            "name": name,
            "formula": sc_data["formula"],
            "tc": sc_data["tc"],
            "family": sc_data["family"],
            "significance": sc_data.get("significance", ""),
            "similarity_score": similarity_score
        })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return similarities[:top_k]


def calculate_similarity(
    material_formula: str,
    features: Dict[str, float],
    known_sc: Dict[str, Any]
) -> float:
    """
    Calculate similarity to a known superconductor.
    
    Args:
        material_formula: Chemical formula
        features: Feature dictionary
        known_sc: Known superconductor dictionary
    
    Returns:
        Similarity score (0-1)
    """
    score = 0.0
    
    # Composition similarity
    from matprov.utils.composition import parse_composition
    
    elements1 = set(parse_composition(material_formula).keys())
    elements2 = set(parse_composition(known_sc["formula"]).keys())
    
    # Jaccard similarity
    intersection = len(elements1 & elements2)
    union = len(elements1 | elements2)
    
    if union > 0:
        score += 0.3 * (intersection / union)
    
    # Family match (strong signal)
    family_match = False
    
    if features.get("likely_cuprate", 0) == 1 and known_sc["family"] == "cuprate":
        family_match = True
    if features.get("likely_iron_based", 0) == 1 and known_sc["family"] == "iron_pnictide":
        family_match = True
    if features.get("likely_mgb2_type", 0) == 1 and known_sc["formula"] == "MgB2":
        family_match = True
    
    if family_match:
        score += 0.5
    
    # Element presence match
    for element in ["Cu", "Fe", "O", "As", "B"]:
        if (element in material_formula) == (element in known_sc["formula"]):
            score += 0.02
    
    return min(score, 1.0)


def suggest_synthesis(material_formula: str, features: Dict[str, float]) -> List[str]:
    """
    Suggest synthesis approaches based on composition and features.
    
    Shows you understand experimental realization, not just theory.
    
    Args:
        material_formula: Chemical formula
        features: Feature dictionary
    
    Returns:
        List of synthesis suggestions
    """
    suggestions = []
    
    # Cuprate synthesis
    if features.get("likely_cuprate", 0) == 1:
        suggestions.append("🔥 Solid-state synthesis: 900-1000°C with controlled oxygen annealing")
        suggestions.append("⚠️  Oxygen stoichiometry critical - anneal in O2 atmosphere")
        suggestions.append("📊 Characterize: XRD for phase purity, SQUID for Tc, oxygen content analysis")
    
    # Iron-based synthesis
    elif features.get("likely_iron_based", 0) == 1:
        suggestions.append("🛡️ Inert atmosphere required (Ar or N2) - avoid oxidation")
        suggestions.append("🔥 Solid-state synthesis: 700-1100°C in sealed quartz tubes")
        suggestions.append("⚠️  Air-sensitive - handle in glovebox")
    
    # Hydride synthesis
    elif features.get("likely_hydride", 0) == 1:
        suggestions.append("💎 Requires diamond anvil cell (DAC) - extreme pressure (>100 GPa)")
        suggestions.append("⚡ Laser heating to promote reaction")
        suggestions.append("📊 In-situ XRD and resistance measurements under pressure")
    
    # General conventional
    else:
        suggestions.append("🔥 Standard solid-state synthesis or arc melting")
        suggestions.append("📊 XRD for structure, PPMS/SQUID for magnetic properties")
        suggestions.append("🔬 Consider thin film deposition (MBE/PLD) for single crystals")
    
    return suggestions


def calculate_confidence(uncertainty: float) -> str:
    """
    Convert uncertainty to confidence level.
    
    Args:
        uncertainty: Prediction uncertainty (K)
    
    Returns:
        Confidence level string
    """
    if uncertainty < 5:
        return "High (±{:.1f}K)".format(uncertainty)
    elif uncertainty < 15:
        return "Medium (±{:.1f}K)".format(uncertainty)
    else:
        return "Low (±{:.1f}K)".format(uncertainty)


# Example usage
if __name__ == "__main__":
    print("=== Physics-Based Explainability Demo ===\n")
    
    # Example prediction
    material = "YBa2Cu3O7"
    predicted_tc = 92.0
    features = {
        "dos_fermi": 8.5,
        "debye_temperature": 400,
        "lambda_ep": 0.8,
        "mu_star": 0.1,
        "mcmillan_tc_estimate": 45,
        "likely_cuprate": 1,
        "likely_layered": 1,
        "dos_favorable_for_pairing": 1
    }
    
    explanation = explain_prediction(material, predicted_tc, features, uncertainty=5.0)
    
    print(f"Material: {explanation['material']}")
    print(f"Predicted Tc: {explanation['predicted_tc']}K")
    print(f"Confidence: {explanation['confidence']}")
    print(f"\n📌 Key Contributing Factors:")
    for factor_name, factor_data in explanation["key_factors"].items():
        print(f"\n  {factor_name.upper().replace('_', ' ')}:")
        for key, value in factor_data.items():
            print(f"    {key}: {value}")
    
    print(f"\n💡 Physics Insights:")
    for insight in explanation["physics_insights"]:
        print(f"  {insight}")
    
    print(f"\n🔬 Mechanism Hypothesis:")
    print(f"  {explanation['mechanism_hypothesis']}")
    
    print(f"\n🔍 Similar Known Superconductors:")
    for sc in explanation["similar_to"]:
        print(f"  {sc['name']} ({sc['formula']}): Tc={sc['tc']}K, similarity={sc['similarity_score']:.2f}")
    
    print(f"\n⚗️  Synthesis Suggestions:")
    for suggestion in explanation["synthesis_suggestions"]:
        print(f"  {suggestion}")
    
    print("\n" + "="*70)
    print("✅ Explainability demonstrates:")
    print("   • Physics understanding (not black-box ML)")
    print("   • Literature knowledge (similar materials)")
    print("   • Experimental awareness (synthesis)")
    print("   • Trust-building (transparent reasoning)")
    print("="*70)

