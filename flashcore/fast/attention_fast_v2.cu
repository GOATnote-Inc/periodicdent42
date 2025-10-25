// Fast Attention V2 - Fix memory access patterns
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

// Use cuBLAS-like approach: better memory hierarchy
__global__ void __launch_bounds__(128)
attention_fast_v2(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int H, int S, int D
) {
    int head = blockIdx.x;
    int q_idx = blockIdx.y * 4 + threadIdx.x / 32;  // 4 queries per block, 4 warps
    int lane = threadIdx.x % 32;
    
    if (q_idx >= S) return;
    
    int offset = head * S * D;
    const half* Q_h = Q + offset;
    const half* K_h = K + offset;
    const half* V_h = V + offset;
    half* O_h = O + offset;
    
    // Load Q row (64 elements, 2 per lane)
    half2 q_vals[2];
    if (lane < 32) {
        q_vals[0] = *((half2*)(Q_h + q_idx*D + lane*2));
        q_vals[1] = *((half2*)(Q_h + q_idx*D + 32 + lane*2));
    }
    
    float scale = rsqrtf(64.0f);
    float row_max = -10000.0f;
    float row_sum = 0.0f;
    
    // Shared for scores
    __shared__ float scores[512];
    
    // Compute scores (each warp does some)
    for (int k = threadIdx.x; k < S; k += 128) {
        float score = 0.0f;
        // Dot product
        for (int d = 0; d < 64; d += 2) {
            half2 k_val = *((half2*)(K_h + k*D + d));
            half2 q_val = (d < 32) ? q_vals[0] : q_vals[1];
            if (d >= 32) {
                int offset_in_pair = (d - 32) / 2;
                if (offset_in_pair < 16) {
                    q_val = q_vals[1];
                    // Adjust for position
                }
            }
            score += __half2float(k_val.x) * __half2float(q_val.x);
            score += __half2float(k_val.y) * __half2float(q_val.y);
        }
        scores[k] = score * scale;
        row_max = fmaxf(row_max, scores[k]);
    }
    __syncthreads();
    
    // Warp reduce max
    for (int offset = 16; offset > 0; offset /= 2) {
        row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
    }
    if (lane == 0) scores[0] = row_max;
    __syncthreads();
    row_max = scores[0];
    
    // Softmax
    for (int k = threadIdx.x; k < S; k += 128) {
        float e = __expf(scores[k] - row_max);
        scores[k] = e;
        row_sum += e;
    }
    __syncthreads();
    
    // Reduce sum
    for (int offset = 16; offset > 0; offset /= 2) {
        row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
    }
    if (lane == 0) scores[1] = row_sum;
    __syncthreads();
    row_sum = 1.0f / scores[1];
    
    // Normalize
    for (int k = threadIdx.x; k < S; k += 128) {
        scores[k] *= row_sum;
    }
    __syncthreads();
    
    // Output = scores @ V (each thread does some dims)
    float out[64] = {0};
    for (int k = 0; k < S; k++) {
        float p = scores[k];
        for (int d = threadIdx.x; d < 64; d += 128) {
            out[d] += p * __half2float(V_h[k*D + d]);
        }
    }
    
    // Write (coalesced)
    for (int d = threadIdx.x; d < 64; d += 128) {
        O_h[q_idx*D + d] = __float2half(out[d]);
    }
}

void bench(const half* Q, const half* K, const half* V, half* O, int H, int S, int D) {
    dim3 grid(H, (S+3)/4);
    dim3 block(128);
    
    for (int i = 0; i < 100; i++) attention_fast_v2<<<grid, block>>>(Q,K,V,O,H,S,D);
    cudaDeviceSynchronize();
    
    std::vector<float> t;
    for (int i = 0; i < 1000; i++) {
        cudaEvent_t s,e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        attention_fast_v2<<<grid, block>>>(Q,K,V,O,H,S,D);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms,s,e);
        t.push_back(ms);
        cudaEventDestroy(s); cudaEventDestroy(e);
    }
    std::sort(t.begin(), t.end());
    float med = t[t.size()/2]*1000;
    printf("V2: %.2f μs\n", med);
    FILE* f=fopen("v2.txt","w"); fprintf(f,"%.2f\n",med); fclose(f);
}

int main() {
    int H=8, S=512, D=64;
    size_t bytes = H*S*D*sizeof(half);
    half *d_Q,*d_K,*d_V,*d_O;
    cudaMalloc(&d_Q,bytes); cudaMalloc(&d_K,bytes);
    cudaMalloc(&d_V,bytes); cudaMalloc(&d_O,bytes);
    half* h=(half*)malloc(bytes);
    for(size_t i=0;i<H*S*D;i++) h[i]=__float2half(rand()/(float)RAND_MAX);
    cudaMemcpy(d_Q,h,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_K,h,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_V,h,bytes,cudaMemcpyHostToDevice);
    free(h);
    bench(d_Q,d_K,d_V,d_O,H,S,D);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}

