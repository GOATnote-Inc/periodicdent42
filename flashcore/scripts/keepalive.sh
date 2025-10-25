#!/usr/bin/env bash
# =============================================================================
# GPU Keep-Alive for FlashCore
# =============================================================================
# Starts a tmux session with nvidia-smi dmon logger to prevent GPU idle
# Usage: bash scripts/keepalive.sh

set -euo pipefail

SESSION="gpu"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/gpu_dmon.log"

# Create logs directory
mkdir -p "$LOG_DIR"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "[ERROR] tmux not found - install with: apt-get install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[INFO] tmux session '$SESSION' already exists"
    echo "[INFO] Attach with: tmux attach -t $SESSION"
    echo "[INFO] Kill with: tmux kill-session -t $SESSION"
else
    # Create new detached session
    tmux new-session -d -s "$SESSION"
    
    # Send command to start nvidia-smi dmon logger
    # -s pucm: power, utilization, clock, memory
    # -d 5: sample every 5 seconds
    tmux send-keys -t "$SESSION" "nvidia-smi dmon -s pucm -d 5 > $LOG_FILE 2>&1" C-m
    
    echo "[OK] tmux session '$SESSION' created and running"
    echo "[OK] GPU monitoring started → $LOG_FILE"
    echo ""
    echo "Commands:"
    echo "  View log:    tail -f $LOG_FILE"
    echo "  Attach:      tmux attach -t $SESSION"
    echo "  Detach:      Ctrl+B, then D"
    echo "  Kill:        tmux kill-session -t $SESSION"
fi

