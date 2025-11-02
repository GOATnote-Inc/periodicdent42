# Landing Page Audit & Cleanup Plan

## Current State Analysis

### Root Directory Files/Folders

| Item | Type | Status | Action |
|------|------|--------|--------|
| **README.md** | File | ✅ Essential | KEEP - Main landing page |
| **LICENSE** | File | ✅ Essential | KEEP - Legal requirement |
| **BlackwellSparseK/** | Dir | ✅ Core project | KEEP - Main work |
| CONTRIBUTING.md | File | ✅ Good | KEEP - Standard for OSS |
| CUTLASS_PR_CHECKLIST.md | File | ⚠️ Status doc | MOVE to docs/ |
| Dockerfile | File | ✅ Good | KEEP - Reproducibility |
| Justfile | File | ⚠️ Minimal | EVALUATE |
| Makefile | File | ⚠️ Minimal | EVALUATE |
| OPEN_SOURCE_COMPLETE.md | File | ❌ Status doc | MOVE to docs/ |
| RELEASE_READY.md | File | ❌ Status doc | MOVE to docs/ |
| config/ | Dir | ⚠️ Lock files | EVALUATE |
| csrc/ | Dir | ⚠️ Legacy | EVALUATE/ARCHIVE |
| docs/ | Dir | ✅ Good | KEEP |
| examples/ | Dir | ⚠️ Mixed | EVALUATE |
| python/ | Dir | ⚠️ Legacy (triageattention) | ARCHIVE |
| pyproject.toml | File | ✅ Essential | KEEP |
| scripts/ | Dir | ⚠️ Mixed | EVALUATE |
| setup.py | File | ✅ Essential | KEEP |
| tests/ | Dir | ✅ Good | KEEP (if relevant) |
| tools/ | Dir | ⚠️ Legacy | EVALUATE/ARCHIVE |

## Professional Repository Standards

### What Top Repos Have

**NVIDIA CUTLASS:**
```
README.md
LICENSE.txt
CMakeLists.txt
include/
examples/
test/
tools/
media/
docs/
CONTRIBUTING.md
```

**PyTorch:**
```
README.md
LICENSE
setup.py
torch/
test/
docs/
examples/
CONTRIBUTING.md
.github/
```

**FlashAttention:**
```
README.md
setup.py
flash_attn/
csrc/
hopper/
tests/
benchmarks/
```

### Common Pattern
- Clean root with essential files only
- One main project directory
- docs/ for documentation
- tests/ for tests
- examples/ for examples
- No status documents on landing page
- No legacy/unrelated directories

## Proposed Cleanup

### Phase 1: Move Status Docs (IMMEDIATE)
```bash
# Move status documents to docs/
mv CUTLASS_PR_CHECKLIST.md docs/project/
mv OPEN_SOURCE_COMPLETE.md docs/project/
mv RELEASE_READY.md docs/project/
```

### Phase 2: Archive Legacy Code (IMMEDIATE)
```bash
# Archive old TriageAttention code
mkdir -p .archive/legacy_projects/
mv python/triageattention .archive/legacy_projects/
mv csrc/ .archive/legacy_projects/  # If not used for current GEMM

# Archive unrelated tools
mv tools/ .archive/legacy_projects/  # If not used for current GEMM
```

### Phase 3: Clean Examples (IF NEEDED)
```bash
# Check if examples/ is relevant to BlackwellSparseK
# If not:
mv examples/ .archive/legacy_projects/
```

### Phase 4: Consolidate Build Files (OPTIONAL)
```bash
# If Justfile/Makefile are minimal and unused:
mv Justfile .archive/
mv Makefile .archive/
# Keep Dockerfile (good for reproducibility)
```

### Phase 5: Update README (FINAL)
Update root README.md to reflect clean structure

## Recommended Final Structure

```
periodicdent42/
├── README.md                  # Landing page (clean, minimal)
├── LICENSE                    # BSD 3-Clause
├── CONTRIBUTING.md            # How to contribute
├── Dockerfile                 # Reproducibility
├── setup.py                   # Python package
├── pyproject.toml             # Python config
├── BlackwellSparseK/          # Main project
│   ├── README.md              # Project docs
│   ├── src/                   # CUDA kernels
│   ├── examples/              # Usage examples
│   ├── benchmarks/            # Performance tests
│   └── docs/                  # Detailed docs
├── docs/                      # Repository-wide docs
│   ├── project/               # Status docs
│   └── development/           # Dev guides
├── tests/                     # Tests (if any)
└── .archive/                  # Legacy code
    └── legacy_projects/       # Old work
```

## Benefits of Cleanup

### Before (Current)
- 20+ items in root directory
- Mix of current and legacy
- Status documents prominent
- Confusing for newcomers
- Not professional appearance

### After (Proposed)
- ~10 essential items in root
- Clear project focus
- Professional appearance
- Easy for newcomers to navigate
- Matches industry standards (CUTLASS, PyTorch, FA3)

## Comparison: Before vs After

### Before (Current Root)
```
BlackwellSparseK/          ✅
CONTRIBUTING.md            ✅
CUTLASS_PR_CHECKLIST.md    ⚠️ (status doc)
Dockerfile                 ✅
Justfile                   ⚠️ (minimal)
LICENSE                    ✅
Makefile                   ⚠️ (minimal)
OPEN_SOURCE_COMPLETE.md    ❌ (status doc)
README.md                  ✅
RELEASE_READY.md           ❌ (status doc)
config/                    ⚠️ (lock files)
csrc/                      ⚠️ (legacy?)
docs/                      ✅
examples/                  ⚠️ (mixed)
python/                    ❌ (legacy triageattention)
pyproject.toml             ✅
scripts/                   ⚠️ (mixed)
setup.py                   ✅
tests/                     ⚠️ (relevant?)
tools/                     ❌ (legacy)
```

### After (Proposed Root)
```
README.md                  ✅ Clean, professional
LICENSE                    ✅ Essential
CONTRIBUTING.md            ✅ OSS standard
Dockerfile                 ✅ Reproducibility
setup.py                   ✅ Python package
pyproject.toml             ✅ Python config
BlackwellSparseK/          ✅ Main project
docs/                      ✅ Documentation
tests/                     ✅ If relevant
.archive/                  ✅ Hidden legacy
```

**Reduction:** 20+ items → 10 essential items

## Implementation Order

### Priority 1: Status Docs (5 minutes)
```bash
mkdir -p docs/project
mv CUTLASS_PR_CHECKLIST.md docs/project/
mv OPEN_SOURCE_COMPLETE.md docs/project/
mv RELEASE_READY.md docs/project/
```

### Priority 2: Legacy Code (10 minutes)
```bash
mkdir -p .archive/legacy_projects
mv python/triageattention .archive/legacy_projects/
# Check if csrc/, tools/, examples/ are legacy
# Move if not related to current GEMM work
```

### Priority 3: Optional Cleanup (5 minutes)
```bash
# If Justfile/Makefile unused:
mv Justfile .archive/ 2>/dev/null
mv Makefile .archive/ 2>/dev/null

# If config/ is just lock files:
mv config/ .archive/ 2>/dev/null
```

### Priority 4: Update README (10 minutes)
- Update root README.md to reflect new structure
- Add clear project structure section
- Link to docs/ for additional information

## Success Criteria

### Measurements
- **File count in root:** <15 items
- **Clarity:** Immediate understanding of project focus
- **Professional:** Matches CUTLASS/PyTorch standards
- **Navigation:** Easy to find main code, docs, examples
- **Legacy:** Hidden but preserved in .archive/

### Visual Test
First-time visitor sees:
1. Clear README explaining project (GEMM optimization)
2. Obvious main directory (BlackwellSparseK/)
3. Standard OSS files (LICENSE, CONTRIBUTING)
4. No confusion from legacy/status documents
5. Professional, focused appearance

## Execution

Ready to execute cleanup in 4 phases:

1. **Status docs → docs/project/** (IMMEDIATE)
2. **Legacy code → .archive/** (IMMEDIATE)
3. **Optional cleanup** (AS NEEDED)
4. **README update** (FINAL)

**Estimated time:** 30 minutes  
**Risk:** Low (everything preserved in .archive/)  
**Benefit:** Professional appearance matching industry standards

---

**Ready to execute?** Say "clean up landing page" to proceed.

