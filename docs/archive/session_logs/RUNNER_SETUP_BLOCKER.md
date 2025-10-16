# Runner Setup: Network Connectivity Blocker

## Date
2025-10-13

## Issue
GPU instance (cudadent42-l4-dev) cannot connect to GitHub.com to download the Actions runner.

**Error:**
```
curl: (28) Failed to connect to github.com port 443 after 133118 ms: Connection timed out
```

## Root Cause
The GPU instance lacks external internet connectivity. This was working during earlier sessions but may be due to:
1. Instance was recreated without external IP
2. VPC network configuration changed
3. Cloud NAT not configured
4. Firewall rules blocking egress

## Solutions

### Option A: Download Runner Locally and Upload (5 min)
**Fastest approach - bypasses network issue**

```bash
# On local machine
cd /Users/kiteboard/periodicdent42
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Upload to GPU instance
gcloud compute scp actions-runner-linux-x64-2.311.0.tar.gz \
  cudadent42-l4-dev:~/actions-runner.tar.gz \
  --zone=us-central1-a

# SSH and configure
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# On GPU instance:
mkdir -p ~/actions-runner && cd ~/actions-runner
mv ~/actions-runner.tar.gz ./actions-runner-linux-x64-2.311.0.tar.gz
tar xzf actions-runner-linux-x64-2.311.0.tar.gz

# Configure runner (use your token)
./config.sh \
  --url https://github.com/GOATnote-Inc/periodicdent42 \
  --token BRSUHH2SKDIYWZCLVYJAA4LI5SSSS \
  --name cudadent42-l4-runner \
  --labels self-hosted,gpu,cuda

# Start runner
nohup ./run.sh > runner.log 2>&1 &
exit
```

### Option B: Add External IP to Instance (10 min)
**Requires instance recreation**

```bash
# Stop instance
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a

# Add external IP
gcloud compute instances add-access-config cudadent42-l4-dev \
  --zone=us-central1-a \
  --access-config-name="External NAT"

# Start instance
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a

# Then follow original runner setup
```

### Option C: Configure Cloud NAT (15 min)
**Provides external connectivity without public IPs**

```bash
# Create Cloud Router
gcloud compute routers create nat-router \
  --network default \
  --region us-central1

# Create NAT gateway
gcloud compute routers nats create nat-config \
  --router=nat-router \
  --region=us-central1 \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges

# Restart instance to pick up new network config
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a

# Then follow original runner setup
```

## Recommended: Option A (Local Download + Upload)
**Reasons:**
- Fastest (5 minutes)
- No infrastructure changes required
- Works immediately
- No cost impact

## Step-by-Step: Option A

### 1. Download Runner Locally (2 min)
```bash
cd /Users/kiteboard/periodicdent42
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Verify download
ls -lh actions-runner-linux-x64-2.311.0.tar.gz
# Should show ~146 MB
```

### 2. Upload to GPU (1 min)
```bash
gcloud compute scp actions-runner-linux-x64-2.311.0.tar.gz \
  cudadent42-l4-dev:~/ \
  --zone=us-central1-a
```

### 3. SSH and Configure (3 min)
```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a
```

**On GPU instance:**
```bash
# Create directory and extract
mkdir -p ~/actions-runner && cd ~/actions-runner
mv ~/actions-runner-linux-x64-2.311.0.tar.gz ./
tar xzf actions-runner-linux-x64-2.311.0.tar.gz

# Configure with your token
./config.sh \
  --url https://github.com/GOATnote-Inc/periodicdent42 \
  --token BRSUHH2SKDIYWZCLVYJAA4LI5SSSS \
  --name cudadent42-l4-runner \
  --labels self-hosted,gpu,cuda

# You'll see prompts:
# Enter the name of the runner group: [Press Enter for default]
# Enter the name of work folder: [Press Enter for _work]

# Start runner in background
nohup ./run.sh > runner.log 2>&1 &

# Verify it's running
ps aux | grep run.sh

# Exit SSH
exit
```

### 4. Verify Runner (1 min)
Visit: https://github.com/GOATnote-Inc/periodicdent42/settings/actions/runners

Should show:
- **Name**: cudadent42-l4-runner
- **Status**: Idle (green dot)
- **Labels**: self-hosted, gpu, cuda

## Next Steps After Runner is Configured

### 1. Create Test PR (2 min)
https://github.com/GOATnote-Inc/periodicdent42/pull/new/test/ci-benchmark-validation

- **Title**: test: Validate CI benchmark workflow
- **Base**: main ← Compare: test/ci-benchmark-validation
- Create pull request

### 2. Add Label (1 min)
On PR: Labels → "benchmark" → Enter

### 3. Watch Workflow (2 min)
Actions tab → CUDA Benchmark → Should run ~30 seconds

### 4. Verify (1 min)
- All steps green
- Download artifacts (results.json, comparison.json)
- `"is_regression": false`

## Troubleshooting

### If upload fails
```bash
# Check instance is running
gcloud compute instances list --filter="name:cudadent42-l4-dev"

# If TERMINATED, start it
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
```

### If runner config fails
```bash
# Check if token expired (tokens expire after 1 hour)
# Generate new token: https://github.com/GOATnote-Inc/periodicdent42/settings/actions/runners/new
```

### If runner doesn't appear as "Idle"
```bash
# SSH to GPU
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# Check runner logs
cd ~/actions-runner
tail -50 runner.log
```

## Cost Note

**GPU is currently RUNNING**
- Cost: $0.20/hour
- Recommendation: Complete runner setup, test workflow, then stop instance

```bash
# After successful test, stop GPU
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a
```

## Status
- [x] Issue identified (network connectivity)
- [x] Solutions documented
- [x] Option A recommended (fastest)
- [ ] Runner download (local machine)
- [ ] Runner upload (gcloud scp)
- [ ] Runner configuration (SSH)
- [ ] Verification (GitHub UI)

## Summary

**Blocker:** GPU instance can't reach GitHub

**Solution:** Download runner locally, upload via gcloud scp, configure via SSH

**Time:** 5 minutes total

**Commands ready to execute above ⬆️**

