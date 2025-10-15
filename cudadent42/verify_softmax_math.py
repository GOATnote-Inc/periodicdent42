#!/usr/bin/env python3
"""
Verify online softmax accumulation math
Simulate V3's online softmax to check if the formula is correct
"""

import numpy as np

# Test case: S=512, BLOCK_N=64 → 8 tiles
np.random.seed(42)

S = 512
BLOCK_N = 64
num_tiles = S // BLOCK_N

# Generate random attention scores (already scaled)
S_full = np.random.randn(S)

# Method 1: Standard softmax (reference)
max_s = S_full.max()
exp_s = np.exp(S_full - max_s)
softmax_ref = exp_s / exp_s.sum()

print("=" * 80)
print("Online Softmax Verification")
print("=" * 80)

print(f"\nReference softmax:")
print(f"  max_s = {max_s:.6f}")
print(f"  sum(exp) = {exp_s.sum():.6f}")
print(f"  softmax sum = {softmax_ref.sum():.6f}")

# Method 2: Online softmax (V3's approach)
m_i = -np.inf
l_i = 0.0

for tile_idx in range(num_tiles):
    start = tile_idx * BLOCK_N
    end = start + BLOCK_N
    S_tile = S_full[start:end]
    
    # Find max
    m_old = m_i
    m_new = max(m_old, S_tile.max())
    
    # Compute correction
    if m_old == -np.inf:
        correction = 1.0
    else:
        correction = np.exp(m_old - m_new)
    
    # Update l_i
    l_new = l_i * correction
    
    # Add new exp values
    for s in S_tile:
        l_new += np.exp(s - m_new)
    
    # Update state
    m_i = m_new
    l_i = l_new
    
    print(f"\nTile {tile_idx}:")
    print(f"  m_old={m_old:.6f}, m_new={m_new:.6f}")
    print(f"  correction={correction:.6f}")
    print(f"  l_i after tile: {l_i:.6f}")

print(f"\n{'=' * 80}")
print(f"Final l_i (online): {l_i:.6f}")
print(f"Final l_i (reference): {exp_s.sum():.6f}")
print(f"Ratio (online/ref): {l_i / exp_s.sum():.6f}")
print(f"{'=' * 80}")

if abs(l_i / exp_s.sum() - 1.0) < 1e-6:
    print("\n✅ Online softmax is mathematically correct!")
else:
    print(f"\n❌ Online softmax is WRONG by {l_i / exp_s.sum():.6f}×")
    print("This confirms a bug in the implementation.")

# Check if the ratio matches our observed 1.48×
if abs(l_i / exp_s.sum() - 1.48) < 0.01:
    print(f"\n⚠️  MATCHES OBSERVED 1.48× BUG!")
    print("The online softmax formula implementation is incorrect.")

