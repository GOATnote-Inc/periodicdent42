# Phase D.3 Debug Log: Hour 7

## Finding: Normalization "Fix" Was Wrong!

**Time**: Hour 7  
**Status**: Reverted incorrect fix

---

### What I Thought:
Line 172 used unnormalized probabilities → added `/l_new`

### Why I Was Wrong:
Flash Attention algorithm:
1. Accumulates **unnormalized** `exp(score - m)` values
2. Only normalizes at **the very end** (`O /= l_final`)
3. This is the CORRECT algorithm!

### Revert:
```cuda
// WRONG "fix":
float p_normalized = S_row[n] / l_new;  ❌

// CORRECT (original):
float p = S_row[n];  ✅ (unnormalized during accumulation)
```

---

## Current Status:

**Still broken**: max_diff=448.0  
**Real bug**: Still unknown

**Hypotheses**:
1. `O_row[]` not initialized per warp?
2. Warp reduction accumulating wrong values?
3. Output writing bug?
4. Something else?

**Next**: Add debug prints to see actual values

---

**Lesson**: Don't "fix" things without understanding the algorithm!


