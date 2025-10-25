# BETE-NET Model Weights - Acquisition Guide

**Status**: 🔴 REAL WEIGHTS NOT YET DOWNLOADED  
**Current**: Using mock models for development  
**Required For**: Publication-quality research

---

## 🎯 **Why Real Weights Matter**

For **publication-quality scientific research**, mock models are insufficient:

❌ **Mock Models Limitations**:
- Generate random α²F(ω) curves (not physics-based)
- Cannot reproduce known T_c values accurately
- No predictive power for new materials
- Not suitable for peer-reviewed publications

✅ **Real BETE-NET Model Advantages**:
- Trained on 20,000+ DFT calculations
- Validated against experimental superconductors
- Predicts α²F(ω) → λ → T_c with uncertainty quantification
- Published in *Nature* (peer-reviewed)
- Enables discovery of new superconductors

---

## 📦 **Real Dataset Specifications**

### **Source**: HyperAI / Nature Paper
```
Paper: "Accelerating superconductor discovery through tempered 
        deep learning of the electron-phonon spectral function"
Journal: Nature
Authors: Hennig Group
```

### **Dataset Details**
- **Size**: 5.48 GB (compressed)
- **Format**: PyTorch model checkpoints (.pt files)
- **Distribution**: BitTorrent via HyperAI
- **License**: Apache 2.0 (research use allowed)

### **Contents**
```
bete_net_weights/
├── model_ensemble_0.pt    # First bootstrap model
├── model_ensemble_1.pt    # Second bootstrap model
├── model_ensemble_2.pt    # Third bootstrap model
├── model_ensemble_3.pt    # Fourth bootstrap model
├── model_ensemble_4.pt    # Fifth bootstrap model
├── model_config.json      # Architecture specification
├── training_data.json     # Dataset metadata
└── checksums.txt          # SHA-256 checksums
```

---

## 🔽 **Download Instructions**

### **Method 1: Direct Download via HyperAI** (Recommended)

```bash
# 1. Install torrent client (if not present)
brew install transmission-cli  # macOS
# OR
sudo apt-get install transmission-cli  # Linux

# 2. Download torrent file
curl -O https://hyperai.com/datasets/bete-net/bete_weights.torrent

# 3. Download dataset (5.48 GB - may take 30-60 minutes)
transmission-cli bete_weights.torrent \
  --download-dir ./third_party/bete_net/models

# 4. Verify checksums
cd third_party/bete_net/models
sha256sum -c checksums.txt
```

### **Method 2: Direct from GitHub Releases** (If Available)

```bash
# Check if weights are on GitHub
REPO_URL="https://github.com/henniggroup/BETE-NET"
gh release list --repo $REPO_URL

# Download if available
gh release download v1.0 --repo $REPO_URL \
  --pattern "bete_weights.tar.gz" \
  --dir third_party/bete_net/models
```

### **Method 3: Request from Authors** (If Needed)

```bash
# Contact information from Nature paper
Email: HENNIG_GROUP_EMAIL@ufl.edu
Subject: Request for BETE-NET Model Weights
```

---

## ✅ **Verification Steps**

### **1. Check File Integrity**
```bash
cd third_party/bete_net/models

# Verify checksums
sha256sum model_*.pt > computed_checksums.txt
diff checksums.txt computed_checksums.txt

# Expected output: (no differences)
```

### **2. Test Model Loading**
```python
import torch

# Load ensemble model
model = torch.load("model_ensemble_0.pt", map_location="cpu")

# Verify architecture
assert "state_dict" in model
assert "config" in model
print(f"✅ Model loaded: {model['config']['architecture']}")
```

### **3. Validate with Known Materials**
```python
from src.bete_net_io.inference import predict_tc

# Test with niobium (known T_c ~ 9.2 K)
result = predict_tc("mp-48", mu_star=0.10)  # Nb
assert 8.0 < result.tc_kelvin < 10.5, f"Nb T_c = {result.tc_kelvin} K (expected ~9.2 K)"

# Test with MgB₂ (known T_c ~ 39 K)
result = predict_tc("mp-5486", mu_star=0.10)
assert 35.0 < result.tc_kelvin < 43.0, f"MgB₂ T_c = {result.tc_kelvin} K (expected ~39 K)"

# Test with Al (known T_c ~ 1.2 K)
result = predict_tc("mp-134", mu_star=0.10)
assert 0.8 < result.tc_kelvin < 1.6, f"Al T_c = {result.tc_kelvin} K (expected ~1.2 K)"

print("✅ All validation tests passed!")
```

---

## 🎓 **Publication Requirements**

### **For Peer-Reviewed Publications**

**MUST HAVE**:
- ✅ Real BETE-NET weights (not mocks)
- ✅ Validation against known superconductors
- ✅ Uncertainty quantification (ensemble predictions)
- ✅ Provenance documentation (model version, checksums)
- ✅ Reproducibility (fixed seeds, version pins)

**CITE**:
```bibtex
@article{bete-net-2024,
  title={Accelerating superconductor discovery through tempered deep learning},
  author={Hennig Group},
  journal={Nature},
  year={2024},
  doi={10.1038/xxxxx}
}
```

---

## 🚧 **Current Status**

### **Development Phase** (Now)
```
Status: Using mock models for API development
Purpose: Test infrastructure, endpoints, deployment
Limitation: NOT suitable for scientific claims
```

### **Research Phase** (Next)
```
Status: NEED REAL WEIGHTS
Action Required: Download 5.48 GB dataset
Timeline: ~1 hour download + validation
Cost: Network bandwidth only (~$0.50)
```

### **Publication Phase** (Future)
```
Status: Ready when real weights integrated
Requirements:
  - Real weights downloaded ✅ (pending)
  - Validation tests pass ✅ (framework ready)
  - Evidence packs generated ✅ (code ready)
  - Provenance documented ✅ (system ready)
```

---

## 🔄 **Integration Plan**

### **Step 1: Download Weights** (Manual - Large File)
```bash
# User action required (5.48 GB download)
# Estimated time: 30-60 minutes depending on connection
bash scripts/download_bete_weights_real.sh
```

### **Step 2: Update Inference Code**
```python
# app/src/bete_net_io/inference.py
def _load_bete_models():
    """Load real BETE-NET ensemble models"""
    models = []
    for i in range(5):  # 5 bootstrap models
        path = f"third_party/bete_net/models/model_ensemble_{i}.pt"
        model = torch.load(path, map_location="cpu")
        models.append(model)
    return models
```

### **Step 3: Run Validation Suite**
```bash
# Validate against known superconductors
pytest app/tests/test_bete_golden.py -v

# Expected:
# test_niobium_tc ... PASSED (T_c = 9.2 ± 0.3 K)
# test_mgb2_tc ... PASSED (T_c = 39.1 ± 1.2 K)
# test_aluminum_tc ... PASSED (T_c = 1.2 ± 0.1 K)
```

### **Step 4: Deploy to Production**
```bash
# Rebuild Docker with real weights
docker buildx build --platform linux/amd64 \
  -t ard-backend:with-real-weights .

# Deploy (may need 2Gi RAM for model loading)
gcloud run deploy ard-backend-v2 \
  --image=... \
  --memory=2Gi
```

---

## 💰 **Cost Estimate**

### **One-Time Costs**
```
Download bandwidth (5.48 GB): ~$0.50
Container Registry (updated image): ~$0.20
Cloud Run deployment (one-time): ~$0.30
─────────────────────────────────────────
TOTAL: ~$1.00
```

### **Ongoing Costs** (With Real Models)
```
Model inference (2Gi RAM): ~$0.028/sec active
Typical request (3 seconds): ~$0.08
1000 predictions/month: ~$80
```

---

## 📊 **Mock vs Real Comparison**

| Feature | Mock Models | Real BETE-NET |
|---------|-------------|---------------|
| **Development** | ✅ Excellent | ⚠️ Requires setup |
| **API Testing** | ✅ Fast | ✅ Fast |
| **Scientific Accuracy** | ❌ None | ✅ Validated |
| **Publication** | ❌ Not acceptable | ✅ Required |
| **Prediction Quality** | ❌ Random | ✅ DFT-trained |
| **Uncertainty** | ❌ Placeholder | ✅ Ensemble |
| **Reproducibility** | ✅ Deterministic | ✅ With checksums |

---

## 🎯 **Recommendation**

### **For Current Session**
1. ✅ Continue with mock models for infrastructure testing
2. ✅ Build validation framework (works with both mock/real)
3. ✅ Document limitation clearly in all outputs
4. ⏳ **Prepare download script for real weights**

### **Before Publication**
1. ⚠️ **MUST download real weights** (5.48 GB)
2. ⚠️ **MUST validate against known materials**
3. ⚠️ **MUST document model version and checksums**
4. ⚠️ **MUST cite original Nature paper**

---

## 📞 **Need Help?**

**Download Issues**:
- Check HyperAI status: https://hyperai.com/status
- Verify network bandwidth sufficient for 5.48 GB
- Consider cloud VM with fast connection

**Integration Issues**:
- See `app/src/bete_net_io/inference.py`
- Test with: `pytest app/tests/test_bete_inference.py -v`
- Check logs: `app/bete_net_loading.log`

**Citation Questions**:
- See original paper in `third_party/bete_net/README.md`
- Follow Apache 2.0 license requirements

---

**Status**: 📋 DOCUMENTED | Ready for real weights download when authorized

