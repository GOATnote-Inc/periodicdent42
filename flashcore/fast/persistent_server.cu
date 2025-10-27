// Persistent GPU server - eliminate launch overhead
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <atomic>

std::atomic<long> total_queries(0);

__global__ void softmax_kernel(half* data, int* offsets, int MB, int B, int groups) {
    int rb = blockIdx.x;
    if (rb >= MB) return;
    
    int start = offsets[rb];
    int end = offsets[rb+1];
    
    __shared__ float smax, ssum;
    if (threadIdx.x == 0) { smax = -1e9f; ssum = 0.0f; }
    __syncthreads();
    
    float lmax = -1e9f;
    for (int t = start; t < end; t++) {
        half* tile = data + t * B * B;
        for (int i = threadIdx.x; i < B*B; i += blockDim.x) {
            lmax = fmaxf(lmax, __half2float(tile[i]));
        }
    }
    atomicMax((int*)&smax, __float_as_int(lmax));
    __syncthreads();
    
    float lsum = 0.0f;
    for (int t = start; t < end; t++) {
        half* tile = data + t * B * B;
        for (int i = threadIdx.x; i < B*B; i += blockDim.x) {
            float v = expf(__half2float(tile[i]) - smax);
            tile[i] = __float2half(v);
            lsum += v;
        }
    }
    atomicAdd(&ssum, lsum);
    __syncthreads();
    
    for (int t = start; t < end; t++) {
        half* tile = data + t * B * B;
        for (int i = threadIdx.x; i < B*B; i += blockDim.x) {
            tile[i] = __float2half(__half2float(tile[i]) / ssum);
        }
    }
}

struct WorkerArgs {
    int id;
    int groups;
    half* d_data;
    int* d_offsets;
    int MB;
    volatile bool* running;
};

void* worker_thread(void* arg) {
    WorkerArgs* args = (WorkerArgs*)arg;
    cudaSetDevice(0);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    while (*args->running) {
        // Simulate inference
        softmax_kernel<<<args->MB, 256, 0, stream>>>(
            args->d_data, args->d_offsets, args->MB, 128, args->groups);
        cudaStreamSynchronize(stream);
        
        total_queries++;
        usleep(500); // 0.5ms between launches
    }
    
    cudaStreamDestroy(stream);
    return nullptr;
}

int main(int argc, char** argv) {
    int num_workers = argc > 1 ? atoi(argv[1]) : 8;
    int duration_sec = argc > 2 ? atoi(argv[2]) : 30;
    int groups = 800;
    
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("PERSISTENT GPU SERVER\n");
    printf("Workers: %d, Duration: %ds, Groups: %d\n", num_workers, duration_sec, groups);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
    
    const int B = 128;
    int MB = (int)sqrtf(groups / 0.2f);
    
    // Allocate GPU memory
    half *d_data;
    int *d_offsets;
    cudaMalloc(&d_data, groups * B * B * sizeof(half));
    cudaMalloc(&d_offsets, (MB + 1) * sizeof(int));
    
    int h_offsets[MB + 1];
    for (int i = 0; i <= MB; i++) h_offsets[i] = (i * groups) / MB;
    cudaMemcpy(d_offsets, h_offsets, (MB+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_data, 0, groups * B * B * sizeof(half));
    
    printf("✅ GPU initialized\n\n");
    
    volatile bool running = true;
    
    // Launch workers
    pthread_t threads[num_workers];
    WorkerArgs args[num_workers];
    
    time_t start_time = time(NULL);
    
    for (int i = 0; i < num_workers; i++) {
        args[i] = {i, groups, d_data, d_offsets, MB, &running};
        pthread_create(&threads[i], nullptr, worker_thread, &args[i]);
    }
    
    // Monitor
    for (int t = 0; t < duration_sec; t++) {
        sleep(1);
        float qps = total_queries.load() / (float)(t + 1);
        printf("  %2ds: %ld queries, %.1f QPS\n", t + 1, total_queries.load(), qps);
    }
    
    running = false;
    
    for (int i = 0; i < num_workers; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    time_t end_time = time(NULL);
    long elapsed_sec = end_time - start_time;
    
    long queries = total_queries.load();
    float qps = queries / (float)elapsed_sec;
    
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("RESULTS\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
    printf("Total queries:  %ld\n", queries);
    printf("Duration:       %lds\n", elapsed_sec);
    printf("QPS:            %.1f\n", qps);
    printf("Target (>200):  %s\n\n", qps > 200 ? "✅" : "❌");
    
    cudaFree(d_data);
    cudaFree(d_offsets);
    
    return 0;
}
