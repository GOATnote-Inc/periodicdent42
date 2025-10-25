#!/bin/bash
set -euo pipefail
IP="${1:-154.57.34.90}"
PORT="${2:-36088}"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes"

ssh -p $PORT $SSH_OPTS root@$IP "mkdir -p /w/flash_v1"
scp -P $PORT $SSH_OPTS flashcore/fast/attention_flash_v1.cu root@$IP:/w/flash_v1/

ssh -p $PORT $SSH_OPTS root@$IP 'cd /w/flash_v1 && export PATH=/usr/local/cuda/bin:$PATH && nvcc -O3 --use_fast_math -arch=sm_90 attention_flash_v1.cu -o flash_v1 && ./flash_v1 && python3 -c "
import torch, torch.nn.functional as F
Q=torch.randn(1,8,512,64,device=\"cuda\",dtype=torch.float16)
K,V=Q.clone(),Q.clone()
for _ in range(100): F.scaled_dot_product_attention(Q,K,V,is_causal=False)
torch.cuda.synchronize()
t=[]
for _ in range(1000):
  s=torch.cuda.Event(enable_timing=True)
  e=torch.cuda.Event(enable_timing=True)
  s.record()
  F.scaled_dot_product_attention(Q,K,V,is_causal=False)
  e.record()
  torch.cuda.synchronize()
  t.append(s.elapsed_time(e)*1000)
t.sort()
print(f\"\\nPyTorch SDPA: {t[len(t)//2]:.2f} μs\")
with open(\"flash_v1_perf.txt\") as f: k=float(f.read().split(\"=\")[1])
print(f\"Flash V1:     {k:.2f} μs\")
print(f\"Speedup:      {t[len(t)//2]/k:.2f}×\")
print(f\"Target 5×:    {t[len(t)//2]/5:.2f} μs\")
"'

scp -P $PORT $SSH_OPTS root@$IP:/w/flash_v1/flash_v1_perf.txt . 2>/dev/null || true
