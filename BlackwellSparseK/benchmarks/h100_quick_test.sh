#!/bin/bash
# Quick H100 benchmark script for RunPod Web Terminal
# Usage: Paste this entire script into RunPod Web Terminal
set -e
cd /tmp
echo "🚀 H100 BlackwellSparseK Benchmark"
apt-get update -qq && apt-get install -y git >/dev/null 2>&1
if [ ! -d "/opt/cutlass" ]; then
  git clone --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass
fi
nvcc --version | grep "release"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
cat > bench.cu << 'KERNEL'
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <vector>
#define BM 512
#define BN 128
#define BK 112
__global__ void bsr_spmm(const half *A,const int *Ar,const int *Ac,int Mb,int Kb,const half *B,const int *Br,const int *Bc,int Nb,float *C,int M,int N){using namespace nvcuda;int bm=blockIdx.x,bn=blockIdx.y;if(bm>=Mb||bn>=Nb)return;__shared__ half sA[BM*BK],sB[BK*BN];wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.0f);for(int ai=Ar[bm];ai<Ar[bm+1];ai++){int kb=Ac[ai];for(int bi=Br[kb];bi<Br[kb+1];bi++){if(Bc[bi]!=bn)continue;for(int i=threadIdx.x;i<BM*BK/8;i+=blockDim.x)((float4*)&sA[(i/(BK/8))*BK+(i%(BK/8))*8])[0]=((float4*)&A[ai*BM*BK+(i/(BK/8))*BK+(i%(BK/8))*8])[0];for(int i=threadIdx.x;i<BK*BN/8;i+=blockDim.x)((float4*)&sB[(i/(BN/8))*BN+(i%(BN/8))*8])[0]=((float4*)&B[bi*BK*BN+(i/(BN/8))*BN+(i%(BN/8))*8])[0];__syncthreads();wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa;wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb;for(int k=0;k<BK;k+=16){wmma::load_matrix_sync(fa,&sA[k],BK);wmma::load_matrix_sync(fb,&sB[k*BN],BN);wmma::mma_sync(acc,fa,fb,acc);}__syncthreads();}}if(bm*BM<M&&bn*BN<N)wmma::store_matrix_sync(&C[bm*BM*N+bn*BN],acc,N,wmma::mem_row_major);}
int main(){const int M=8192,N=8192,K=8192,Mb=16,Nb=64,Kb=74,tk=16;printf("Matrix: %dx%dx%d\n",M,N,K);std::vector<int> Ar(Mb+1,0),Br(Kb+1,0),Ac,Bc;srand(42);for(int i=0;i<Mb;i++){for(int j=0;j<tk;j++)Ac.push_back(rand()%Kb);Ar[i+1]=Ac.size();}for(int i=0;i<Kb;i++){for(int j=0;j<tk;j++)Bc.push_back(rand()%Nb);Br[i+1]=Bc.size();}int nA=Ac.size(),nB=Bc.size();printf("nnzb_A=%d nnzb_B=%d\n",nA,nB);half *dA,*dB;float *dC;int *dAr,*dAc,*dBr,*dBc;cudaMalloc(&dA,nA*BM*BK*sizeof(half));cudaMalloc(&dB,nB*BK*BN*sizeof(half));cudaMalloc(&dC,M*N*sizeof(float));cudaMalloc(&dAr,(Mb+1)*sizeof(int));cudaMalloc(&dAc,nA*sizeof(int));cudaMalloc(&dBr,(Kb+1)*sizeof(int));cudaMalloc(&dBc,nB*sizeof(int));std::vector<half> hA(nA*BM*BK),hB(nB*BK*BN);for(auto& x:hA)x=__float2half(0.1f);for(auto& x:hB)x=__float2half(0.1f);cudaMemcpy(dA,hA.data(),nA*BM*BK*sizeof(half),cudaMemcpyHostToDevice);cudaMemcpy(dB,hB.data(),nB*BK*BN*sizeof(half),cudaMemcpyHostToDevice);cudaMemcpy(dAr,Ar.data(),(Mb+1)*sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(dAc,Ac.data(),nA*sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(dBr,Br.data(),(Kb+1)*sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(dBc,Bc.data(),nB*sizeof(int),cudaMemcpyHostToDevice);cudaMemset(dC,0,M*N*sizeof(float));dim3 g(Mb,Nb),b(256);bsr_spmm<<<g,b>>>(dA,dAr,dAc,Mb,Kb,dB,dBr,dBc,Nb,dC,M,N);cudaDeviceSynchronize();cudaError_t e=cudaGetLastError();if(e!=cudaSuccess){printf("CUDA ERROR: %s\n",cudaGetErrorString(e));return 1;}printf("✅ Kernel launched successfully\n");cudaEvent_t s,t;cudaEventCreate(&s);cudaEventCreate(&t);cudaEventRecord(s);for(int i=0;i<100;i++)bsr_spmm<<<g,b>>>(dA,dAr,dAc,Mb,Kb,dB,dBr,dBc,Nb,dC,M,N);cudaEventRecord(t);cudaEventSynchronize(t);float ms;cudaEventElapsedTime(&ms,s,t);float avg_ms=ms/100;float tflops=(2LL*M*N*K/1e12)/(avg_ms/1000);printf("\n╔═══════════════════════════════╗\n");printf("║  H100 BENCHMARK RESULTS       ║\n");printf("╠═══════════════════════════════╣\n");printf("║  Time:    %.3f ms           ║\n",avg_ms);printf("║  TFLOPS:  %.1f               ║\n",tflops);printf("╚═══════════════════════════════╝\n");return 0;}
KERNEL
echo ""
echo "Compiling for H100 (sm_90a)..."
nvcc -O3 --use_fast_math -std=c++17 -arch=sm_90a bench.cu -o bench -lcudart
echo "✅ Compiled"
echo ""
./bench

