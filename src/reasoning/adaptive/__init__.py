"""
Adaptive optimization router - EXPERIMENTAL.

This module explores whether automatically selecting between Bayesian Optimization
and RL based on estimated noise levels improves performance.

## Status: Research Prototype

This is NOT production-ready. We're testing a hypothesis:
- Hypothesis: RL may be more robust than BO at high noise levels
- Evidence: Preliminary validation on Branin function (n=10 trials, noise=2.0)
- Limitation: Single test function, limited noise levels, no real hardware

## What We Need to Validate:
1. Does this hold across multiple test functions?
2. What is the actual noise threshold where RL becomes advantageous?
3. Does this work on real experiments (not just simulations)?
4. Is noise estimation reliable enough to make routing decisions?

## Use with Caution:
- For research and prototyping only
- Gather more validation data before production use
- Document all assumptions and limitations
- Be transparent about confidence levels
"""

__version__ = "0.1.0-alpha"
__status__ = "Experimental"

