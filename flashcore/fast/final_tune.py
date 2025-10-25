import torch
import triton
import triton.language as tl

@triton.jit
def attn(Q,K,V,O, sh,sm,sk, H,N,D, BM:tl.constexpr,BN:tl.constexpr):
    pid,off_h = tl.program_id(0),tl.program_id(1)
    om,on,od = pid*BM+tl.arange(0,BM),tl.arange(0,BN),tl.arange(0,64)
    q = tl.load(Q+off_h*sh+om[:,None]*sm+od[None,:]*sk, mask=om[:,None]<N)
    m,l,acc = tl.zeros([BM],dtype=tl.float32)-1e10, tl.zeros([BM],dtype=tl.float32), tl.zeros([BM,64],dtype=tl.float32)
    for ns in range(0,N,BN):
        k = tl.load(K+off_h*sh+(ns+on)[:,None]*sm+od[None,:]*sk, mask=(ns+on[:,None])<N)
        v = tl.load(V+off_h*sh+(ns+on)[:,None]*sm+od[None,:]*sk, mask=(ns+on[:,None])<N)
        s = tl.dot(q,tl.trans(k))*0.125
        mn = tl.maximum(m,tl.max(s,1))
        p = tl.exp(s-mn[:,None])
        ln = tl.exp(m-mn)*l + tl.sum(p,1)
        acc = acc*tl.exp(m-mn)[:,None]*(l/ln)[:,None] + tl.dot(p.to(v.dtype),v)/ln[:,None]
        m,l = mn,ln
    tl.store(O+off_h*sh+om[:,None]*sm+od[None,:]*sk, acc, mask=om[:,None]<N)

def run(q,k,v,BM,BN):
    _,H,N,_=q.shape
    o=torch.empty_like(q)
    attn[((N+BM-1)//BM,H)](q,k,v,o,q.stride(1),q.stride(2),q.stride(3),H,N,64,BM,BN)
    return o

best_us,best_cfg=999,None
for BM in [32,64,128]:
    for BN in [32,64,128]:
        try:
            q=torch.randn(1,8,512,64,device="cuda",dtype=torch.float16)
            for _ in range(20): run(q,q,q,BM,BN)
            torch.cuda.synchronize()
            t=[]
            for _ in range(200):
                s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                s.record(); run(q,q,q,BM,BN); e.record(); torch.cuda.synchronize()
                t.append(s.elapsed_time(e)*1000)
            t.sort()
            med=t[len(t)//2]
            if med<best_us:
                best_us,best_cfg=med,(BM,BN)
                print(f"{BM:3}x{BN:3}: {med:5.1f}us BEST")
            else:
                print(f"{BM:3}x{BN:3}: {med:5.1f}us")
        except: print(f"{BM:3}x{BN:3}: FAIL")

print(f"\nOptimal: {best_cfg[0]}x{best_cfg[1]} = {best_us:.1f}us")

q=torch.randn(1,8,512,64,device="cuda",dtype=torch.float16)
for _ in range(50): torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=False)
torch.cuda.synchronize()
t=[]
for _ in range(300):
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record(); torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=False); e.record()
    torch.cuda.synchronize(); t.append(s.elapsed_time(e)*1000)
t.sort()
sdpa=t[len(t)//2]
target=sdpa/5
print(f"SDPA: {sdpa:.1f}us")
print(f"Target (5x): {target:.1f}us")
print(f"Gap: {best_us/target:.1f}x away from target")
print(f"vs SDPA: {best_us/sdpa:.2f}x slower")

