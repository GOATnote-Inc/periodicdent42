# GitHub Actions Runner Setup

## Required for CI Workflow

The CUDA benchmark workflow requires a self-hosted runner with GPU access.

## Setup Instructions

### 1. Generate Runner Token

Visit: https://github.com/GOATnote-Inc/periodicdent42/settings/actions/runners/new

Click "New self-hosted runner" → Copy the token

### 2. Install Runner on GPU Instance

```bash
# SSH to GPU instance
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# Create runner directory
mkdir -p ~/actions-runner && cd ~/actions-runner

# Download runner (Linux x64)
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# Extract
tar xzf actions-runner-linux-x64-2.311.0.tar.gz

# Configure (use token from step 1)
./config.sh \
  --url https://github.com/GOATnote-Inc/periodicdent42 \
  --token YOUR_TOKEN_HERE \
  --name cudadent42-l4-runner \
  --labels self-hosted,gpu,cuda \
  --work _work

# Install as service (optional, for persistence)
sudo ./svc.sh install
sudo ./svc.sh start

# Or run in foreground for testing
./run.sh
```

### 3. Verify Runner

Check: https://github.com/GOATnote-Inc/periodicdent42/settings/actions/runners

Should show "cudadent42-l4-runner" as "Idle"

## Testing Workflow

### Create Test PR

```bash
# Local machine
cd /Users/kiteboard/periodicdent42
git checkout -b test/ci-benchmark-validation
echo "Test CI workflow" > CI_TEST.md
git add CI_TEST.md
git commit -m "test: Trigger CI benchmark workflow"
git push -u origin test/ci-benchmark-validation
```

### Trigger Workflow

1. Open PR on GitHub
2. Add label "benchmark"
3. Watch Actions tab
4. Verify:
   - Workflow triggers
   - Build succeeds
   - Benchmark runs
   - Comparison completes
   - Artifacts uploaded

## Troubleshooting

### Runner Not Appearing

```bash
# Check runner status
cd ~/actions-runner
./run.sh --check

# Check service status (if installed)
sudo ./svc.sh status

# View logs
journalctl -u actions.runner.*
```

### Workflow Not Triggering

- Verify PR has label "benchmark"
- Check workflow file paths in `.github/workflows/cuda_benchmark.yml`
- Ensure runner has label "self-hosted"
- Check runner is "Idle" not "Offline"

### Build Fails

```bash
# On GPU instance, test manually
cd periodicdent42/cudadent42
python setup.py build_ext --inplace

# Check CUDA
nvidia-smi
nvcc --version
```

### Benchmark Fails

```bash
# Test manually
cd periodicdent42/cudadent42/bench
python3 integrated_test.py --output test.json
```

## Cost Management

L4 GPU: $0.20/hour

**Recommendation:** Stop instance when not actively using runner

```bash
# Stop runner service
sudo ./svc.sh stop

# Stop instance
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a

# Start when needed
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
```

## Security

- Runner has access to repository secrets
- Runs in isolated directory (`_work/`)
- Does not have sudo access (unless explicitly granted)
- Limited to labeled workflows only

## Uninstall

```bash
# Stop runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall

# Remove from GitHub (via web UI)
# Delete runner directory
cd ~ && rm -rf actions-runner
```

