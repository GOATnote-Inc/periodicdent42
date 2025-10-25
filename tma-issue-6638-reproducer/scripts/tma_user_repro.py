import os, torch, triton
import triton.language as tl

# ---- Kernel: block-pointer path (Triton 3.0.x) ----
@triton.jit
def copy_tile_bp(X, Y,
                 M: tl.constexpr, N: tl.constexpr,
                 STRIDE_XM, STRIDE_XN,
                 STRIDE_YM, STRIDE_YN,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    base_m = pid_m * BLOCK_M

    # three expert iterations → different offsets (descriptor-in-loop semantics)
    for expert in range(3):
        off_m = base_m + expert * 7  # force a tail occasionally
        # rebuild block pointers **inside the loop**
        x_blk = tl.make_block_ptr(base=X,
                                  shape=(M, N),
                                  strides=(STRIDE_XM, STRIDE_XN),
                                  offsets=(off_m, 0),
                                  block_shape=(BLOCK_M, BLOCK_N),
                                  order=(1, 0))
        y_blk = tl.make_block_ptr(base=Y,
                                  shape=(M, N),
                                  strides=(STRIDE_YM, STRIDE_YN),
                                  offsets=(off_m, 0),
                                  block_shape=(BLOCK_M, BLOCK_N),
                                  order=(1, 0))
        # boundary checks for tails (🔒 security)
        a = tl.load(x_blk, boundary_check=(0,1))
        # predicated store on even experts only
        if (expert & 1) == 0:
            tl.store(y_blk, a, boundary_check=(0,1))

def main():
    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.float16
    M, N = 1024, 512              # tile-friendly, with potential tails
    X = torch.randn((M, N), device=dev, dtype=dtype)
    Y = torch.zeros_like(X)

    grid = (triton.cdiv(M, 128),)
    copy_tile_bp[grid](
        X, Y, M, N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_M=128, BLOCK_N=128,
        num_stages=4, num_warps=8
    )
    torch.cuda.synchronize()

    # Optional: 1000x determinism smoke-test (fast; deep test driven separately)
    ok = True
    ref = Y.clone()
    for _ in range(32):  # keep it quick here
        Y.zero_()
        copy_tile_bp[grid](X, Y, M, N, X.stride(0), X.stride(1), Y.stride(0), Y.stride(1),
                           BLOCK_M=128, BLOCK_N=128, num_stages=4, num_warps=8)
        torch.cuda.synchronize()
        ok &= torch.equal(ref, Y)
    print("Determinism(32x):", "PASS" if ok else "FAIL")

if __name__ == "__main__":
    # Keep TRITON dumps tight and discoverable
    os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
    os.environ.setdefault("TRITON_KERNEL_DUMP_DIR", "./artifacts/tmp")
    main()

