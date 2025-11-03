#!/bin/bash
# Persistent Brev authentication script
# Expert GPU dev strategy: SSH keypair + tmux for infinite sessions

set -e

BREV_TOKEN="${BREV_TOKEN:-$1}"
INSTANCE="${2:-awesome-gpu-name}"

echo "🔐 Setting up persistent Brev access..."
echo ""

# Step 1: Login with fresh token
if [ -n "$BREV_TOKEN" ]; then
    echo "1️⃣ Authenticating with Brev..."
    brev login --token "$BREV_TOKEN"
    echo "✅ Logged in"
else
    echo "✅ Already logged in (or will use existing credentials)"
fi

# Step 2: Generate SSH key if needed
if [ ! -f ~/.ssh/brev_h100 ]; then
    echo ""
    echo "2️⃣ Generating dedicated SSH keypair..."
    ssh-keygen -t ed25519 -f ~/.ssh/brev_h100 -N "" -C "brev-h100-persistent"
    echo "✅ SSH keypair created: ~/.ssh/brev_h100"
else
    echo ""
    echo "2️⃣ SSH keypair already exists"
fi

# Step 3: Add key to Brev
echo ""
echo "3️⃣ Adding SSH key to Brev..."
if brev keys add ~/.ssh/brev_h100.pub 2>/dev/null; then
    echo "✅ SSH key added to Brev"
else
    echo "⚠️  Key may already be added (continuing...)"
fi

# Step 4: Get instance IP
echo ""
echo "4️⃣ Getting instance connection info..."
INSTANCE_INFO=$(brev ls 2>/dev/null | grep "$INSTANCE" || echo "")
if [ -z "$INSTANCE_INFO" ]; then
    echo "⚠️  Instance not found or not ready"
    echo "   Run: brev ls"
    echo "   Then retry with correct instance name"
    exit 1
fi

echo "✅ Instance found: $INSTANCE"

# Step 5: Create persistent SSH config
echo ""
echo "5️⃣ Creating SSH config..."
cat > ~/.ssh/brev_config << SSHEOF
Host brev-h100
    HostName \$(brev ls | grep $INSTANCE | awk '{print \$4}')
    User ubuntu
    IdentityFile ~/.ssh/brev_h100
    StrictHostKeyChecking no
    ServerAliveInterval 60
    ServerAliveCountMax 10
    TCPKeepAlive yes
SSHEOF

echo "✅ SSH config created"

# Step 6: Create connection helper script
echo ""
echo "6️⃣ Creating connection helpers..."
cat > ~/.local/bin/brev-connect << 'HELPEREOF'
#!/bin/bash
# Quick connect to Brev H100 with tmux
INSTANCE="${1:-awesome-gpu-name}"
IP=$(brev ls 2>/dev/null | grep "$INSTANCE" | awk '{print $4}')
if [ -z "$IP" ]; then
    echo "❌ Instance not found. Run: brev ls"
    exit 1
fi
echo "🔗 Connecting to $INSTANCE ($IP)..."
ssh -i ~/.ssh/brev_h100 -o StrictHostKeyChecking=no ubuntu@$IP -t 'tmux attach -t ncu || tmux new -s ncu'
HELPEREOF

chmod +x ~/.local/bin/brev-connect

echo "✅ Helper script created: brev-connect"

# Step 7: Test connection
echo ""
echo "7️⃣ Testing connection..."
if brev shell "$INSTANCE" -- echo "Connection test successful" 2>/dev/null; then
    echo "✅ Connection test passed"
else
    echo "⚠️  Direct connection test failed (may need to wait for instance)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ PERSISTENT ACCESS CONFIGURED!"
echo ""
echo "Usage:"
echo "  Quick connect:    brev-connect"
echo "  Direct SSH:       brev shell $INSTANCE"
echo "  With our script:  brev shell $INSTANCE"
echo ""
echo "Session will persist via tmux even if token expires!"
echo "═══════════════════════════════════════════════════════════"

