#!/usr/bin/env python3
import torch
import triton
import triton.language as tl

@triton.jit
def _attn(Q,K,V,O, sh,sm,sk, kh,kn,kd, vh,vn,vd, oh,om,od, H,N,D, BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
    pid_m,pid_h = tl.program_id(0),tl.program_id(1)
    om,on,od = pid_m*BM+tl.arange(0,BM), tl.arange(0,BN), tl.arange(0,BD)
    q = tl.load(Q+pid_h*sh+om[:,None]*sm+od[None,:]*sk, mask=(om[:,None]<N)&(od[None,:]<D), other=0.0)
    m,l,acc = tl.zeros([BM],dtype=tl.float32)-float("inf"), tl.zeros([BM],dtype=tl.float32), tl.zeros([BM,BD],dtype=tl.float32)
    for sn in range(0,(N+BN-1)//BN*BN,BN):
        k = tl.load(K+pid_h*kh+on[:,None]*kn+od[None,:]*kd, mask=(on[:,None]<N-sn)&(od[None,:]<D), other=0.0)
        v = tl.load(V+pid_h*vh+on[:,None]*vn+od[None,:]*vd, mask=(on[:,None]<N-sn)&(od[None,:]<D), other=0.0)
        s = tl.dot(q,tl.trans(k))*(1.0/8.0)
        m_new = tl.maximum(m,tl.max(s,1))
        alpha = tl.exp(m-m_new)
        p = tl.exp(s-m_new[:,None])
        l = l*alpha+tl.sum(p,1)
        acc = acc*alpha[:,None]+tl.dot(p.to(v.dtype),v)
        m = m_new
    acc = acc/l[:,None]
    tl.store(O+pid_h*oh+om[:,None]*om+od[None,:]*od, acc, mask=(om[:,None]<N)&(od[None,:]<D))

def flash(q,k,v,BM,BN):
    B,H,N,D=q.shape
    o=torch.empty_like(q)
    _attn[(triton.cdiv(N,BM),H*B)](q,k,v,o, q.stride(1),q.stride(2),q.stride(3), k.stride(1),k.stride(2),k.stride(3), v.stride(1),v.stride(2),v.stride(3), o.stride(1),o.stride(2),o.stride(3), H,N,D, BM,BN,D)
    return o

def bench(BM,BN):
    q=torch.randn(1,8,512,64,device='cuda',dtype=torch.float16)
    k,v=q.clone(),q.clone()
    for _ in range(50): flash(q,k,v,BM,BN)
    torch.cuda.synchronize()
    t=[]
    for _ in range(500):
        s=torch.cuda.Event(enable_timing=True)
        e=torch.cuda.Event(enable_timing=True)
        s.record(); flash(q,k,v,BM,BN); e.record()
        torch.cuda.synchronize()
        t.append(s.elapsed_time(e)*1000)
    t.sort()
    return t[len(t)//2]

if __name__=='__main__':
    best_t,best_cfg = float('inf'), None
    for BM in [32,64,128]:
        for BN in [32,64,128]:
            try:
                t = bench(BM,BN)
                print(f"BM={BM:3d} BN={BN:3d}: {t:6.2f}μs", end='')
                if t < best_t:
                    best_t,best_cfg = t,(BM,BN)
                    print(" ✅")
                else:
                    print()
            except:
                print(f"BM={BM:3d} BN={BN:3d}: FAIL")
    print(f"\nBest: BM={best_cfg[0]} BN={best_cfg[1]} = {best_t:.2f}μs")
    
    # Compare SDPA
    q=torch.randn(1,8,512,64,device='cuda',dtype=torch.float16)
    for _ in range(100): torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=False)
    torch.cuda.synchronize()
    t=[]
    for _ in range(500):
        s=torch.cuda.Event(enable_timing=True)
        e=torch.cuda.Event(enable_timing=True)
        s.record(); torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=False); e.record()
        torch.cuda.synchronize()
        t.append(s.elapsed_time(e)*1000)
    t.sort()
    sdpa_t = t[len(t)//2]
    print(f"SDPA: {sdpa_t:.2f}μs")
    print(f"Speedup: {sdpa_t/best_t:.2f}×")

