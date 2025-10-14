// ============================================================================
// cudax Stream-Ordered Memory Resource (RAII Gateway)
// ============================================================================
// Purpose: Eliminate allocation jitter from benchmarks via stream-ordered alloc
// Target: CUDA 11.2+ with memory pool support
// ============================================================================

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace cudax {

// ============================================================================
// Stream-Ordered Device Memory Resource
// ============================================================================

class StreamOrderedMR {
public:
    StreamOrderedMR(cudaStream_t stream = nullptr) : stream_(stream) {
        // Use default stream if not specified
        if (stream_ == nullptr) {
            cudaStreamCreate(&stream_);
            owns_stream_ = true;
        }
    }
    
    ~StreamOrderedMR() {
        if (owns_stream_ && stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Allocate from stream-ordered pool
    void* allocate(size_t bytes) {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocAsync(&ptr, bytes, stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMallocAsync failed: ") + cudaGetErrorString(err)
            );
        }
        return ptr;
    }
    
    // Deallocate to stream-ordered pool
    void deallocate(void* ptr) {
        if (ptr) {
            cudaFreeAsync(ptr, stream_);
        }
    }
    
    cudaStream_t stream() const { return stream_; }
    
private:
    cudaStream_t stream_ = nullptr;
    bool owns_stream_ = false;
};

// ============================================================================
// RAII Device Memory Handle
// ============================================================================

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer(size_t count, StreamOrderedMR& mr) 
        : mr_(mr), count_(count), bytes_(count * sizeof(T)) {
        ptr_ = static_cast<T*>(mr_.allocate(bytes_));
    }
    
    ~DeviceBuffer() {
        if (ptr_) {
            mr_.deallocate(ptr_);
        }
    }
    
    // No copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : mr_(other.mr_), ptr_(other.ptr_), count_(other.count_), bytes_(other.bytes_) {
        other.ptr_ = nullptr;
    }
    
    T* get() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return bytes_; }
    
private:
    StreamOrderedMR& mr_;
    T* ptr_ = nullptr;
    size_t count_ = 0;
    size_t bytes_ = 0;
};

} // namespace cudax
