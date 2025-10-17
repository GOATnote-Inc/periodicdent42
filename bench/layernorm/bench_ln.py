import torch, time, os
from bench.layernorm.build_ln import build_ln
torch.backends.cuda.matmul.allow_tf32=True

mod = build_ln(extra=[
    f"-DTHREADS={os.environ.get('THREADS','256')}",
    f"-DROWS_PER_CTA={os.environ.get('ROWS_PER_CTA','1')}",
    f"-DVEC_WIDTH={os.environ.get('VEC_WIDTH','4')}",
    f"-DUSE_WARP={os.environ.get('USE_WARP','1')}",
])

B,H,S,D = 1,8,512,64
x = torch.randn(B,H,S,D, device='cuda', dtype=torch.float16)
gamma = torch.randn(D, device='cuda', dtype=torch.float16)
beta  = torch.randn(D, device='cuda', dtype=torch.float16)

def run_kernel():
    return mod.forward(x, gamma, beta,
                       int(os.environ.get("THREADS","256")),
                       int(os.environ.get("ROWS_PER_CTA","1")),
                       int(os.environ.get("VEC_WIDTH","4")),
                       int(os.environ.get("USE_WARP","1")))

# correctness
with torch.no_grad():
    y0 = run_kernel()
    y1 = torch.nn.functional.layer_norm(x.float(), (D,), gamma.float(), beta.float(), eps=1e-5).half()
    md = (y0.float()-y1.float()).abs().max().item()
    print(f"max_diff={md:.6f}")

# timing
torch.cuda.synchronize()
for _ in range(10): run_kernel()
torch.cuda.synchronize()
t0=time.perf_counter()
for _ in range(100): run_kernel()
torch.cuda.synchronize(); t1=time.perf_counter()
print(f"{(t1-t0)*1e6/100:.2f}")

