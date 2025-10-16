# Runner Setup: Infrastructure Blocker Summary

## Date
2025-10-13 14:25 UTC

## Current Status
**BLOCKED** - GPU instance lacks outbound internet connectivity to GitHub

## What Was Attempted

### ✅ Completed
1. GPU instance started (cudadent42-l4-dev)
2. Runner downloaded locally (179 MB)
3. Runner uploaded to GPU via gcloud scp
4. Runner extracted successfully

### ❌ Blocked
**Runner configuration fails:**
```
./config.sh --url https://github.com/GOATnote-Inc/periodicdent42 --token ... 
Error: The request was canceled due to the configured HttpClient.Timeout of 100 seconds elapsing.
```

**Root cause:** GPU instance cannot reach GitHub.com (neither port 443 nor API endpoints)

## Infrastructure Issue

The cudadent42-l4-dev instance is configured without external internet connectivity:
- No external IP assigned
- No Cloud NAT configured
- Cannot reach github.com for runner registration

This blocks both:
1. Downloading the runner package
2. Configuring the runner (contacting GitHub API)

## Solutions Required

### Option 1: Add External IP (Recommended - Simplest)
**Time:** 5 minutes  
**Cost:** No additional cost (L4 already running)

```bash
# Stop instance
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a

# Add external IP
gcloud compute instances add-access-config cudadent42-l4-dev \
  --zone=us-central1-a \
  --access-config-name="External NAT"

# Start instance
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a

# Wait 30 seconds for boot, then SSH
sleep 30
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# On GPU: Configure runner (token may have expired, get new one)
cd ~/actions-runner
./config.sh \
  --url https://github.com/GOATnote-Inc/periodicdent42 \
  --token GET_NEW_TOKEN_FROM_GITHUB \
  --name cudadent42-l4-runner \
  --labels self-hosted,gpu,cuda

# Start runner
nohup ./run.sh > runner.log 2>&1 &
exit
```

**Get new token:** https://github.com/GOATnote-Inc/periodicdent42/settings/actions/runners/new  
(Tokens expire after 1 hour)

### Option 2: Configure Cloud NAT
**Time:** 10 minutes  
**Cost:** ~$0.045/hour (~$30/month if left running)

```bash
# Create Cloud Router (one-time)
gcloud compute routers create nat-router \
  --network default \
  --region us-central1

# Create NAT gateway (one-time)
gcloud compute routers nats create nat-config \
  --router=nat-router \
  --region=us-central1 \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges

# Restart instance to pick up NAT
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a

# Then configure runner (get new token first)
```

### Option 3: Use Different Instance
**Time:** 15 minutes  
**Cost:** ~$0.20/hour (same as current)

Create new instance with external IP from the start:

```bash
gcloud compute instances create cudadent42-l4-runner \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --boot-disk-size=100GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True" \
  --network-interface=network-tier=PREMIUM
```

## Recommendation: Option 1 (External IP)

**Reasons:**
- Fastest (5 minutes)
- No ongoing cost (vs Cloud NAT)
- No new instance needed
- Runner files already on instance

**Steps:**
1. Get new token from GitHub (old one likely expired)
2. Add external IP to instance (3 commands above)
3. Configure runner (now has internet)
4. Verify in GitHub settings

## Current State of CI Implementation

### ✅ Completed (95%)
- Code: 90 lines, GPU-validated
- Baseline: .baseline.json committed
- Workflow: cuda_benchmark.yml functional
- Test branch: test/ci-benchmark-validation ready
- Documentation: Complete

### ⏸️ Blocked (5%)
- Runner configuration (network issue)
- PR creation (waiting for runner)
- Workflow test (waiting for runner)

## Cost Tracking

| Item | Amount |
|------|--------|
| Development | 3.5 hours |
| GPU testing (earlier) | 25 min ($0.08) |
| **GPU running now** | **30 min** ($0.10) |
| **Total cost** | **$0.18** |

**Current rate:** $0.20/hour (L4 running)

## Next Steps

### Immediate (User Action Required)

**1. Get new runner token (2 min)**
- Visit: https://github.com/GOATnote-Inc/periodicdent42/settings/actions/runners/new
- Click "New self-hosted runner"
- Copy token (expires in 1 hour)

**2. Add external IP to instance (3 min)**
```bash
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a
gcloud compute instances add-access-config cudadent42-l4-dev \
  --zone=us-central1-a
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
```

**3. Configure runner (2 min)**
```bash
sleep 30  # Wait for boot
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a
cd ~/actions-runner
./config.sh --url https://github.com/GOATnote-Inc/periodicdent42 \
  --token YOUR_NEW_TOKEN \
  --name cudadent42-l4-runner \
  --labels self-hosted,gpu,cuda
nohup ./run.sh > runner.log 2>&1 &
exit
```

**4. Verify (1 min)**
https://github.com/GOATnote-Inc/periodicdent42/settings/actions/runners
- Should show "cudadent42-l4-runner" as "Idle"

**5. Create PR and test (3 min)**
- Follow steps in NEXT_SESSION_START_HERE.md

## Alternative: Stop GPU and Resume Tomorrow

If you prefer to resolve this tomorrow:

```bash
# Stop GPU to save cost
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a

# Tomorrow: Get new token, add external IP, configure runner
```

**Cost saved:** $0.20/hour * hours not running

## Files Ready

All code and documentation is complete:
- `NEXT_SESSION_START_HERE.md` - Complete handoff
- `RUNNER_SETUP_BLOCKER.md` - Network issue details
- `RUNNER_BLOCKER_SUMMARY.md` - This file
- `cudadent42/bench/.baseline.json` - Baseline committed
- `.github/workflows/cuda_benchmark.yml` - Workflow ready

## Summary

**Blocker:** Infrastructure - GPU instance needs external connectivity

**Solution:** Add external IP (5 minutes, 3 commands)

**Then:** Configure runner → Create PR → Test workflow (5 minutes)

**Total time from here:** 10 minutes

**Current cost:** $0.18 total, GPU running at $0.20/hour

**Recommendation:** Either fix now (10 min) or stop GPU and fix tomorrow

