#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <cmath>

#define BM 128
#define BN 128
#define BK 64
#define ELEMS_PER_THREAD ((BM*BN)/256) // 64 elements per thread

__global__ void __launch_bounds__(256) bsr_registers(
    half const* __restrict__ A_blocks, int const* __restrict__ A_row_ptr, int const* __restrict__ A_col_idx,
    half const* __restrict__ B, float* __restrict__ C, int M, int N, int K, int Mb, int Nb, int Kb) {
    
    int bm = blockIdx.x, bn = blockIdx.y;
    if(bm >= Mb || bn >= Nb) return;
    
    __shared__ half sA[BM*BK], sB[BK*BN];
    
    // Each thread accumulates ELEMS_PER_THREAD outputs in registers
    float acc[ELEMS_PER_THREAD];
    #pragma unroll
    for(int i=0; i<ELEMS_PER_THREAD; i++) acc[i] = 0.0f;
    
    int tid = threadIdx.x;
    
    // Process sparse blocks
    for(int idx = A_row_ptr[bm]; idx < A_row_ptr[bm+1]; idx++) {
        int kb = A_col_idx[idx];
        
        // Load A and B
        for(int i=tid; i<BM*BK; i+=256) sA[i] = A_blocks[idx*BM*BK + i];
        for(int i=tid; i<BK*BN; i+=256) {
            int row = i / BN, col = i % BN;
            sB[i] = (kb*BK + row < K && bn*BN + col < N) ? B[(kb*BK+row)*N + (bn*BN+col)] : __float2half(0.0f);
        }
        __syncthreads();
        
        // Each thread computes its assigned elements
        #pragma unroll 4
        for(int e=0; e<ELEMS_PER_THREAD; e++) {
            int elem = tid + e*256;
            if(elem < BM*BN) {
                int i = elem / BN, j = elem % BN;
                float sum = 0.0f;
                #pragma unroll 8
                for(int k=0; k<BK; k++) {
                    sum += __half2float(sA[i*BK + k]) * __half2float(sB[k*BN + j]);
                }
                acc[e] += sum;
            }
        }
        __syncthreads();
    }
    
    // Write accumulated results
    #pragma unroll
    for(int e=0; e<ELEMS_PER_THREAD; e++) {
        int elem = tid + e*256;
        if(elem < BM*BN) {
            int i = elem / BN, j = elem % BN;
            if(bm*BM + i < M && bn*BN + j < N) {
                C[(bm*BM+i)*N + (bn*BN+j)] = acc[e];
            }
        }
    }
}

// CPU reference
void cpu_bsr(const std::vector<half>& A_blocks, const std::vector<int>& A_row_ptr, 
             const std::vector<int>& A_col_idx, const std::vector<half>& B,
             std::vector<float>& C, int M, int N, int K, int Mb, int Nb) {
    std::fill(C.begin(), C.end(), 0.0f);
    for(int bm = 0; bm < Mb; bm++) {
        for(int bn = 0; bn < Nb; bn++) {
            for(int idx = A_row_ptr[bm]; idx < A_row_ptr[bm+1]; idx++) {
                int kb = A_col_idx[idx];
                for(int i = 0; i < BM && bm*BM+i < M; i++) {
                    for(int j = 0; j < BN && bn*BN+j < N; j++) {
                        float sum = 0.0f;
                        for(int k = 0; k < BK && kb*BK+k < K; k++) {
                            sum += __half2float(A_blocks[idx*BM*BK + i*BK + k]) * 
                                   __half2float(B[(kb*BK+k)*N + (bn*BN+j)]);
                        }
                        C[(bm*BM+i)*N + (bn*BN+j)] += sum;
                    }
                }
            }
        }
    }
}

int main() {
    printf("\n╔═══════════════════════════════════════════════╗\n");
    printf("║  BSR H100 - Register Accumulation  ║\n");
    printf("╚═══════════════════════════════════════════════╝\n\n");
    
    // Correctness test
    {
        printf("═══ Correctness (512x512) ═══\n");
        int M=512, N=512, K=512, Mb=M/BM, Nb=N/BN, Kb=K/BK;
        
        std::vector<int> A_row_ptr(Mb+1), A_col_idx;
        srand(42);
        for(int i=0; i<Mb; i++) {
            A_col_idx.push_back(rand() % Kb);
            A_row_ptr[i+1] = A_col_idx.size();
        }
        int nnz = A_col_idx.size();
        
        std::vector<half> hA(nnz*BM*BK), hB(K*N);
        for(auto&x:hA) x=__float2half((float)rand()/RAND_MAX);
        for(auto&x:hB) x=__float2half((float)rand()/RAND_MAX);
        
        std::vector<float> cpu_C(M*N);
        cpu_bsr(hA, A_row_ptr, A_col_idx, hB, cpu_C, M, N, K, Mb, Nb);
        
        half *dA, *dB; float *dC; int *dAptr, *dAcol;
        cudaMalloc(&dA, nnz*BM*BK*sizeof(half));
        cudaMalloc(&dB, K*N*sizeof(half));
        cudaMalloc(&dC, M*N*sizeof(float));
        cudaMalloc(&dAptr, (Mb+1)*sizeof(int));
        cudaMalloc(&dAcol, nnz*sizeof(int));
        
        cudaMemcpy(dA, hA.data(), nnz*BM*BK*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), K*N*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dAptr, A_row_ptr.data(), (Mb+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dAcol, A_col_idx.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
        
        bsr_registers<<<dim3(Mb,Nb), 256>>>(dA, dAptr, dAcol, dB, dC, M, N, K, Mb, Nb, Kb);
        cudaDeviceSynchronize();
        
        std::vector<float> gpu_C(M*N);
        cudaMemcpy(gpu_C.data(), dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        
        float max_diff = 0.0f;
        for(int i=0; i<M*N; i++) max_diff = fmax(max_diff, fabs(cpu_C[i] - gpu_C[i]));
        
        printf("  Max error: %.6f %s\n\n", max_diff, max_diff < 0.01f ? "✅" : "❌");
        
        if(max_diff >= 0.01f) return 1;
        
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dAptr); cudaFree(dAcol);
    }
    
    // Performance
    for(int size : {2048, 4096, 8192}) {
        printf("═══ %dx%d ═══\n", size, size);
        
        int M=size, N=size, K=size, Mb=M/BM, Nb=N/BN, Kb=K/BK;
        
        std::vector<int> A_row_ptr(Mb+1), A_col_idx;
        srand(42);
        for(int i=0; i<Mb; i++) {
            for(int j=0; j<std::max(1,Kb/8); j++) A_col_idx.push_back(rand() % Kb);
            A_row_ptr[i+1] = A_col_idx.size();
        }
        int nnz = A_col_idx.size();
        
        std::vector<half> hA(nnz*BM*BK), hB(K*N);
        for(auto&x:hA) x=__float2half((float)rand()/RAND_MAX*0.01f);
        for(auto&x:hB) x=__float2half((float)rand()/RAND_MAX*0.01f);
        
        half *dA, *dB; float *dC; int *dAptr, *dAcol;
        cudaMalloc(&dA, nnz*BM*BK*sizeof(half));
        cudaMalloc(&dB, (size_t)K*N*sizeof(half));
        cudaMalloc(&dC, (size_t)M*N*sizeof(float));
        cudaMalloc(&dAptr, (Mb+1)*sizeof(int));
        cudaMalloc(&dAcol, nnz*sizeof(int));
        
        cudaMemcpy(dA, hA.data(), nnz*BM*BK*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), (size_t)K*N*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(dAptr, A_row_ptr.data(), (Mb+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dAcol, A_col_idx.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
        
        for(int i=0; i<10; i++)
            bsr_registers<<<dim3(Mb,Nb), 256>>>(dA, dAptr, dAcol, dB, dC, M, N, K, Mb, Nb, Kb);
        cudaDeviceSynchronize();
        
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for(int i=0; i<50; i++)
            bsr_registers<<<dim3(Mb,Nb), 256>>>(dA, dAptr, dAcol, dB, dC, M, N, K, Mb, Nb, Kb);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        float avg = ms / 50;
        float tflops = (2.0*M*N*K/1e12) / (avg/1000);
        
        // cuBLAS
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        half *dAd, *dBd; float *dCd;
        cudaMalloc(&dAd, (size_t)M*K*sizeof(half));
        cudaMalloc(&dBd, (size_t)K*N*sizeof(half));
        cudaMalloc(&dCd, (size_t)M*N*sizeof(float));
        
        float alpha=1.0f, beta=0.0f;
        for(int i=0; i<10; i++)
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dBd, CUDA_R_16F, N,
                        dAd, CUDA_R_16F, K, &beta, dCd, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        cudaEventRecord(s);
        for(int i=0; i<50; i++)
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dBd, CUDA_R_16F, N,
                        dAd, CUDA_R_16F, K, &beta, dCd, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        
        cudaEventElapsedTime(&ms, s, e);
        float avg_c = ms / 50;
        float tflops_c = (2.0*M*N*K/1e12) / (avg_c/1000);
        
        printf("  BSR:    %.3f ms | %.1f TFLOPS\n", avg, tflops);
        printf("  cuBLAS: %.3f ms | %.1f TFLOPS\n", avg_c, tflops_c);
        printf("  %%: %.1f%%\n\n", 100.0*tflops/tflops_c);
        
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dAptr); cudaFree(dAcol);
        cudaFree(dAd); cudaFree(dBd); cudaFree(dCd);
        cublasDestroy(h);
    }
    
    return 0;
}
