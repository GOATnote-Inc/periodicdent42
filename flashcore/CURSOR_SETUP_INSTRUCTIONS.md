# Cursor Setup Instructions for FlashCore

**Purpose**: Make FlashCore GPU constraints "sticky" across all Cursor sessions  
**Date**: October 21, 2025  
**Estimated Time**: 5 minutes

---

## 🎯 Goal

Ensure Cursor AI always:
1. Uses L4 GPU (never falls back to CPU)
2. Runs preflight before compile/bench/profile
3. Uses correct CUDA architecture (sm_89)
4. Follows test-driven development workflow
5. Maintains project context across sessions

---

## 📋 Step 1: Add Project Rules (Recommended)

**Cursor now uses Rules instead of legacy .cursorrules**

### Option A: Via Cursor UI (Preferred)

1. Open Cursor
2. Go to **Settings** → **Rules** (or press Cmd+, then search "Rules")
3. Click **"+ New Rule"**
4. Fill in:
   - **Name**: `FlashCore GPU Ground Rules`
   - **Type**: `Always` (applies to all prompts)
   - **Scope**: This project only
   - **Content**: Paste contents of `.cursorrules` file

### Option B: Use Legacy .cursorrules File

We've created `.cursorrules` which Cursor will auto-detect:

```bash
# File already exists at:
# /Users/kiteboard/periodicdent42/flashcore/.cursorrules

# Cursor will automatically load this file
# (but migrate to Rules UI for forward compatibility)
```

---

## 📋 Step 2: Enable Memories (Recommended)

**Memories** help Cursor remember project patterns across sessions.

### How to Enable:

1. Open Cursor Chat
2. When Cursor asks "Should I remember this?", click **"Yes"**
3. Manually create key memories:

**Memory 1: GPU Constraint**
```
Title: Always Use L4 GPU
Content: FlashCore must always use the NVIDIA L4 GPU (GCP instance) for all compilation, testing, benchmarking, and profiling. Never fall back to CPU without explicit permission. Engineering time is more valuable than GPU cost - keep the GPU alive.
```

**Memory 2: Preflight Requirement**
```
Title: Run Preflight Before Heavy Operations
Content: Before any compile, benchmark, or profile operation in FlashCore, always run: bash scripts/preflight.sh. If it fails, auto-fix the environment by sourcing scripts/env_cuda_l4.sh and adjusting PATH/CUDA_HOME/LD_LIBRARY_PATH. Show a summary of changes and re-run preflight until it passes.
```

**Memory 3: Project Phase**
```
Title: FlashCore Project Phase
Content: FlashCore is currently in Phase 1 (WMMA Tensor Core implementation). Phase 0 (baseline) is complete with 1500 µs latency. Target for Phase 1: ~150 µs (10× speedup), Tensor Core utilization ≥50%. Primary project goal is Phase 2: <58 µs (≥15× speedup vs 870 µs baseline).
```

**Memory 4: Test-Driven Development**
```
Title: FlashCore TDD Workflow
Content: Always follow test-driven development: 1) Run tests first (pytest tests/test_correctness.py), 2) Implement code changes (small diffs), 3) Re-run tests, 4) Benchmark (python benchmarks/benchmark_latency.py --shape mission), 5) Profile with NCU, 6) Document results. Keep changes small (50-200 lines).
```

---

## 📋 Step 3: Pin Agent Prompt (Optional but Helpful)

Create a **pinned chat** in Cursor with the agent instructions:

1. Open Cursor Chat
2. Paste the "SYSTEM / GROUND RULES" prompt from the user's instructions
3. Click **"Pin"** button (top right)
4. This keeps the instructions visible and accessible

---

## 📋 Step 4: Add AGENTS.md to Context (Optional)

For advanced users, you can reference AGENTS.md in your prompts:

```
@AGENTS.md Please review the operating manual and proceed with Phase 1 WMMA implementation
```

Cursor will include the full AGENTS.md content in context automatically.

---

## 📋 Step 5: Verify Setup

Test that Cursor remembers the rules:

### Test 1: GPU Awareness
```
Prompt: "What GPU should I use for FlashCore?"
Expected: "NVIDIA L4 GPU (GCP instance), never CPU"
```

### Test 2: Preflight Knowledge
```
Prompt: "What should I do before benchmarking?"
Expected: "Run bash scripts/preflight.sh first"
```

### Test 3: Phase Awareness
```
Prompt: "What phase is FlashCore in?"
Expected: "Phase 1 (WMMA), targeting ~150 µs"
```

---

## 🎓 Best Practices

### Use Agent Mode for Multi-Step Operations

When working on complex tasks (Phase 1 WMMA implementation):

1. Switch to **Agent mode** in Cursor Chat
2. Agent can autonomously:
   - Edit multiple files
   - Run terminal commands
   - Fix errors and iterate
   - Keep context across steps

### Example Agent Prompt:
```
Using Agent mode:

1. Run preflight check (bash scripts/preflight.sh)
2. Create kernels/flashcore_wmma.cu from flashcore_baseline.cu
3. Implement WMMA for Q@K^T (following docs/PHASE1_WMMA_GUIDE.md)
4. Build and test with tiny shape first
5. Report PTXAS stats and test results

Stop if any step fails and show diagnostics.
```

### Keep Context Fresh

At the start of each session:

```
Prompt: "Review AGENTS.md and summarize current FlashCore phase, target metrics, and next steps"
```

This rehydrates context from the operating manual.

---

## 📁 Created Files Summary

All setup files are now in place:

```
flashcore/
├── .cursorrules                      ✅ Legacy rules file (Cursor auto-loads)
├── AGENTS.md                         ✅ Complete operating manual
├── CURSOR_SETUP_INSTRUCTIONS.md     ✅ This file
├── L4_GPU_EXECUTION_GUIDE.md        ✅ L4-specific execution guide
│
├── scripts/
│   ├── env_cuda_l4.sh               ✅ CUDA environment setup
│   ├── preflight.sh                 ✅ GPU validation (run before heavy ops)
│   └── keepalive.sh                 ✅ GPU keep-alive (tmux + dmon)
│
└── docs/
    ├── ARCHITECTURE.md               ✅ Technical design
    ├── GETTING_STARTED.md            ✅ Setup guide
    └── PHASE1_WMMA_GUIDE.md          ✅ WMMA implementation guide
```

---

## 🚀 Quick Start After Setup

Once Cursor is configured:

1. **SSH to L4 GPU** (if not already there):
   ```bash
   gcloud compute ssh <instance> --zone=<zone>
   cd /path/to/flashcore
   ```

2. **Run preflight**:
   ```bash
   bash scripts/preflight.sh
   ```

3. **Ask Cursor to proceed**:
   ```
   Prompt: "Preflight passed. Please review AGENTS.md and begin Phase 1 WMMA implementation following docs/PHASE1_WMMA_GUIDE.md. Start with creating kernels/flashcore_wmma.cu from the baseline."
   ```

4. **Cursor will**:
   - Read the operating manual
   - Understand current phase
   - Follow TDD workflow
   - Implement WMMA step-by-step
   - Test and benchmark after each change

---

## 🔧 Troubleshooting

### Issue: Cursor Doesn't Remember Rules

**Fix**: 
- Verify `.cursorrules` exists and has correct content
- Manually add Rules via Cursor UI (Settings → Rules)
- Create Memories manually (Chat → "Remember this")

### Issue: Cursor Suggests CPU Operations

**Fix**:
- Re-emphasize in prompt: "Use L4 GPU, never CPU"
- Check that Memory "Always Use L4 GPU" is enabled
- Reference AGENTS.md explicitly: `@AGENTS.md`

### Issue: Cursor Skips Preflight

**Fix**:
- Add to every prompt: "First run preflight (bash scripts/preflight.sh)"
- Create Memory: "Run Preflight Before Heavy Operations"
- Use Agent mode which follows rules more strictly

---

## ✅ Setup Checklist

- [ ] `.cursorrules` file exists and contains GPU rules
- [ ] Added "FlashCore GPU Ground Rules" in Cursor Settings → Rules
- [ ] Enabled Memories and created 4 key memories (GPU, Preflight, Phase, TDD)
- [ ] Tested Cursor remembers rules (ask about GPU, preflight, phase)
- [ ] (Optional) Pinned agent prompt in Cursor Chat
- [ ] (Optional) Familiar with Agent mode for multi-step operations
- [ ] Ready to SSH to L4 and run preflight

---

## 📞 Next Steps

1. **Complete this setup** (5 minutes)
2. **SSH to L4 GPU instance**
3. **Run preflight** (`bash scripts/preflight.sh`)
4. **Validate baseline** (2 hours: build + test + bench)
5. **Begin Phase 1 WMMA** (30-40 hours, use Agent mode)

---

**Status**: Setup instructions complete ✅  
**Estimated Setup Time**: 5 minutes  
**Next Action**: Configure Cursor Rules and Memories, then proceed to L4 execution

---

**For detailed L4 execution steps, see**: `L4_GPU_EXECUTION_GUIDE.md`  
**For complete operating manual, see**: `AGENTS.md`  
**For Phase 1 implementation, see**: `docs/PHASE1_WMMA_GUIDE.md`

