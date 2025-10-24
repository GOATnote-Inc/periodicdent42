#!/bin/bash
# CUDA Architect Emergency Fix - Correct the firewall rules NOW
set -e

echo "🚨 CRITICAL: Applying ACTUAL security fixes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PROJECT_ID="periodicdent42"
MY_IP="131.161.225.33"  # Your detected IP from previous run

echo "🔧 Fix 1: Restricting SSH to your IP ($MY_IP/32)..."
gcloud compute firewall-rules update default-allow-ssh \
  --source-ranges="$MY_IP/32" \
  --project="$PROJECT_ID" && \
  echo "✅ SSH restricted to $MY_IP/32" || \
  echo "❌ Failed to update SSH rule"

echo ""
echo "🔧 Fix 2: Deleting RDP rule (not needed for Linux GPU)..."
gcloud compute firewall-rules delete default-allow-rdp \
  --project="$PROJECT_ID" \
  --quiet && \
  echo "✅ RDP rule deleted" || \
  echo "⚠️  RDP rule not found or could not be deleted"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

gcloud compute firewall-rules list \
  --project="$PROJECT_ID" \
  --filter="name:default-allow-ssh OR name:default-allow-rdp" \
  --format="table(name,sourceRanges,allowed[].map().firewall_rule().list())"

echo ""
echo "Expected: SSH shows $MY_IP/32, RDP should not exist"
