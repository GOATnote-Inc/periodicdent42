#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifndef ROWS_PER_CTA
#define ROWS_PER_CTA 1
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef VEC_WIDTH
#define VEC_WIDTH 4   // bytes per lane element pack (use 4 -> uint4, 16B)
#endif
#ifndef USE_WARP
#define USE_WARP 1    // 1: warp shuffles; 0: shared-memory reductions
#endif
#ifndef EPS_F
#define EPS_F 1e-5f
#endif

__device__ __forceinline__ float warp_sum(float v){
    #pragma unroll
    for(int d=16; d>0; d>>=1) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}
__device__ __forceinline__ float warp_max(float v){
    #pragma unroll
    for(int d=16; d>0; d>>=1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, d));
    return v;
}
__device__ __forceinline__ float warp_mean(float v, int n){
    return warp_sum(v) / (float)n;
}

template<bool VEC, bool WARP>
__global__ void layernorm_forward_kernel(
    const half* __restrict__ x,   // [R, D]
    half* __restrict__ y,         // [R, D]
    const half* __restrict__ gamma, // [D] or nullptr
    const half* __restrict__ beta,  // [D] or nullptr
    int R, int D
){
    extern __shared__ float sm[];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;

    for(int r0 = (blockIdx.x*ROWS_PER_CTA); r0 < R && r0 < (blockIdx.x+1)*ROWS_PER_CTA; ++r0){
        const half* xr = x + (size_t)r0*D;
        half*       yr = y + (size_t)r0*D;

        float sum = 0.f, sqsum = 0.f;

        if constexpr (VEC){
            // 16B vectorized loads (uint4); D is multiple of 8 (64)
            const uint4* pv = reinterpret_cast<const uint4*>(xr);
            int vecN = D * sizeof(half) / 16; // # of uint4
            for(int i = tid; i < vecN; i += blockDim.x){
                uint4 u = pv[i];
                half2* h2 = (half2*)&u;
                #pragma unroll
                for(int k=0;k<8;k+=2){
                    float a = __half2float(h2[k>>1].x);
                    float b = __half2float(h2[k>>1].y);
                    sum   += a + b;
                    sqsum += a*a + b*b;
                }
            }
        } else {
            for(int i = tid; i < D; i += blockDim.x){
                float v = __half2float(xr[i]);
                sum   += v;
                sqsum += v*v;
            }
        }

        // reduction to CTA
        if constexpr (WARP){
            sum   = warp_sum(sum);
            sqsum = warp_sum(sqsum);
            if (lane==0){
                sm[wid*2+0] = sum;
                sm[wid*2+1] = sqsum;
            }
            __syncthreads();
            if (tid==0){
                float s=0.f, q=0.f;
                int W = blockDim.x>>5;
                #pragma unroll
                for(int w=0; w<W; ++w){ s += sm[w*2+0]; q += sm[w*2+1]; }
                sm[0]=s; sm[1]=q;
            }
            __syncthreads();
            sum = sm[0]; sqsum = sm[1];
        } else {
            // shared-memory tree reduce
            sm[tid] = sum;
            __syncthreads();
            for(int ofs=blockDim.x>>1; ofs>0; ofs>>=1){
                if(tid<ofs) sm[tid]+=sm[tid+ofs];
                __syncthreads();
            }
            sum = sm[0];
            __syncthreads();
            sm[tid] = sqsum;
            __syncthreads();
            for(int ofs=blockDim.x>>1; ofs>0; ofs>>=1){
                if(tid<ofs) sm[tid]+=sm[tid+ofs];
                __syncthreads();
            }
            sqsum = sm[0];
            __syncthreads();
        }

        float mean = sum / (float)D;
        float var  = fmaxf(sqsum / (float)D - mean*mean, 0.f);
        float inv_std = rsqrtf(var + EPS_F);

        if constexpr (VEC){
            uint4* pyo = reinterpret_cast<uint4*>(yr);
            const uint4* pxi = reinterpret_cast<const uint4*>(xr);
            int vecN = D * sizeof(half) / 16;
            for(int i = tid; i < vecN; i += blockDim.x){
                uint4 u = pxi[i];
                half2* h2 = (half2*)&u;
                half2 out4[4];
                #pragma unroll
                for(int k=0;k<4;k++){
                    float a = __half2float(h2[k].x);
                    float b = __half2float(h2[k].y);
                    a = (a - mean)*inv_std;
                    b = (b - mean)*inv_std;
                    if (gamma){
                        float g0 = __half2float(gamma[i*8 + k*2 + 0]);
                        float g1 = __half2float(gamma[i*8 + k*2 + 1]);
                        a *= g0; b *= g1;
                    }
                    if (beta){
                        float b0 = __half2float(beta[i*8 + k*2 + 0]);
                        float b1 = __half2float(beta[i*8 + k*2 + 1]);
                        a += b0; b += b1;
                    }
                    out4[k] = __halves2half2(__float2half(a), __float2half(b));
                }
                uint4 w; half2* w2 = (half2*)&w;
                #pragma unroll
                for(int k=0;k<4;k++) w2[k] = out4[k];
                pyo[i] = w;
            }
        } else {
            for(int i = tid; i < D; i += blockDim.x){
                float v = (__half2float(xr[i]) - mean)*inv_std;
                if (gamma) v *= __half2float(gamma[i]);
                if (beta)  v += __half2float(beta[i]);
                yr[i] = __float2half(v);
            }
        }
        __syncthreads();
    }
}

extern "C" void layernorm_forward_launcher(
    const half* x, half* y, const half* g, const half* b, int R, int D, cudaStream_t s,
    int threads, int rows_per_cta, int vec_width, int use_warp
){
    dim3 block(threads);
    dim3 grid( (R + rows_per_cta - 1) / rows_per_cta );
    size_t smem = (use_warp? (threads/32)*2*sizeof(float) : threads*sizeof(float));
    if (vec_width>=4){
        if (use_warp) layernorm_forward_kernel<true, true><<<grid,block,smem,s>>>(x,y,g,b,R,D);
        else          layernorm_forward_kernel<true,false><<<grid,block,smem,s>>>(x,y,g,b,R,D);
    } else {
        if (use_warp) layernorm_forward_kernel<false,true><<<grid,block,smem,s>>>(x,y,g,b,R,D);
        else          layernorm_forward_kernel<false,false><<<grid,block,smem,s>>>(x,y,g,b,R,D);
    }
}

