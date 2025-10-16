#!/bin/bash
# Find available L4 GPU regions for g2-standard-4

echo "🔍 Searching for L4 GPU capacity (g2-standard-4 + nvidia-l4)..."
echo ""

ZONES=(
    "southamerica-east1-c"  # São Paulo (Brazil)
    "us-west1-b"            # Oregon
    "us-east4-a"            # Virginia
    "us-west4-a"            # Las Vegas
    "europe-west4-a"        # Netherlands
    "asia-southeast1-b"     # Singapore
)

for zone in "${ZONES[@]}"; do
    echo -n "Testing $zone... "
    
    # Try to get machine type (will fail fast if capacity issue)
    result=$(gcloud compute machine-types describe g2-standard-4 \
        --zone="$zone" 2>&1)
    
    if echo "$result" | grep -q "ZONE_RESOURCE_POOL_EXHAUSTED\|not found"; then
        echo "❌ No capacity"
    else
        echo "✅ AVAILABLE!"
        echo ""
        echo "🎯 Found capacity in: $zone"
        echo ""
        echo "To create instance:"
        echo "gcloud compute instances create cudadent42-l4-$zone \\"
        echo "  --zone=$zone \\"
        echo "  --machine-type=g2-standard-4 \\"
        echo "  --accelerator=type=nvidia-l4,count=1 \\"
        echo "  --image-family=ubuntu-2204-lts \\"
        echo "  --image-project=ubuntu-os-cloud \\"
        echo "  --boot-disk-size=100GB \\"
        echo "  --maintenance-policy=TERMINATE"
        exit 0
    fi
done

echo ""
echo "❌ No L4 capacity found in any tested region"
echo "Try: gcloud compute accelerator-types list | grep nvidia-l4"

