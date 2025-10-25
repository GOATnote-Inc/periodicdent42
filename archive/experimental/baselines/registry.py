"""
Baseline Registry System

Allows clean registration and lookup of attention implementations.
"""

from dataclasses import dataclass
from typing import Callable, Dict

@dataclass
class Baseline:
    name: str
    fn: Callable  # (q,k,v,mask,dropout,causal)->(out)

REGISTRY: Dict[str, Baseline] = {}

def register(name: str):
    """Decorator to register a baseline implementation"""
    def deco(fn):
        REGISTRY[name] = Baseline(name, fn)
        return fn
    return deco

def get(name: str) -> Baseline:
    """Get a registered baseline by name"""
    assert name in REGISTRY, f"Unknown baseline {name}. Available: {list(REGISTRY.keys())}"
    return REGISTRY[name]

def list_available():
    """List all registered baselines"""
    return list(REGISTRY.keys())

