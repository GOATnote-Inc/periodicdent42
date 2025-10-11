#!/bin/bash
# Launch GCE GPU instance for automated benchmark execution
# Instance will run benchmarks and auto-shutdown

set -e

INSTANCE_NAME="cudadent42-bench-$(date +%s)"
ZONE="us-central1-a"
MACHINE_TYPE="g2-standard-4"  # L4 GPU
GPU_TYPE="nvidia-l4"
GPU_COUNT=1
IMAGE_FAMILY="ubuntu-2004-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BUCKET_NAME="periodicdent42-benchmarks"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Launching CUDAdent42 Benchmark Instance"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  Instance: $INSTANCE_NAME"
echo "  Zone: $ZONE"
echo "  Machine: $MACHINE_TYPE"
echo "  GPU: $GPU_COUNT x $GPU_TYPE"
echo "  Mode: Preemptible (cost-optimized)"
echo "  Results: gs://$BUCKET_NAME/cudadent42/"
echo ""

# Ensure GCS bucket exists
echo "Checking GCS bucket..."
gsutil ls "gs://$BUCKET_NAME" &>/dev/null || {
    echo "Creating GCS bucket..."
    gsutil mb -p periodicdent42 -c STANDARD -l us-central1 "gs://$BUCKET_NAME"
}
echo "✅ GCS bucket ready"

# Create instance with local startup script
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Creating GPU instance..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

STARTUP_SCRIPT="$(dirname "$0")/gce_benchmark_startup.sh"

gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-standard \
    --maintenance-policy=TERMINATE \
    --preemptible \
    --scopes=cloud-platform \
    --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
    --metadata=install-nvidia-driver=True

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Instance created: $INSTANCE_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Benchmark execution will take approximately 10-15 minutes."
echo "Instance will auto-shutdown when complete."
echo ""
echo "Monitor progress:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -f /var/log/cuda-benchmark.log'"
echo ""
echo "View serial output:"
echo "  gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Check results (after ~15 minutes):"
echo "  gsutil ls gs://$BUCKET_NAME/cudadent42/"
echo ""
echo "Estimated cost: ~$0.75 USD (15 minutes @ $3.06/hour)"
echo ""

# Wait and check for completion
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ Waiting for benchmark completion..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Polling every 60 seconds for results..."
echo "(Press Ctrl+C to stop waiting, instance will continue)"
echo ""

for i in {1..30}; do
    echo "[$i/30] Checking for results..."
    
    # Check if instance is still running
    STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format="get(status)" 2>/dev/null || echo "NOT_FOUND")
    
    if [ "$STATUS" = "TERMINATED" ]; then
        echo ""
        echo "✅ Instance terminated (benchmark complete or failed)"
        break
    fi
    
    # Check for results in GCS
    if gsutil ls "gs://$BUCKET_NAME/cudadent42/sota_*" &>/dev/null; then
        echo ""
        echo "✅ Results detected in cloud storage!"
        break
    fi
    
    sleep 60
done

# Download results
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📥 Downloading Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

LOCAL_RESULTS_DIR="$(dirname "$0")/../benchmark_results"
mkdir -p "$LOCAL_RESULTS_DIR"

gsutil -m rsync -r "gs://$BUCKET_NAME/cudadent42/" "$LOCAL_RESULTS_DIR/" || {
    echo "⚠️  No results found yet. Check back in a few minutes."
    echo "    gsutil -m rsync -r gs://$BUCKET_NAME/cudadent42/ $LOCAL_RESULTS_DIR/"
    exit 0
}

echo ""
echo "✅ Results downloaded to: $LOCAL_RESULTS_DIR"
echo ""

# Display summary
LATEST_RESULT=$(ls -t "$LOCAL_RESULTS_DIR"/sota_*/benchmark_log.txt | head -1)
if [ -f "$LATEST_RESULT" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Benchmark Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    grep -A 10 "SUMMARY" "$LATEST_RESULT" | head -20 || echo "Summary not found in log"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Benchmark execution complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Review results in: $LOCAL_RESULTS_DIR"
echo "  2. Update README with actual performance numbers"
echo "  3. Commit results to repository"
echo ""

