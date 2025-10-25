// ============================================================================
// CUDA Graph Capture & Replay Wrapper
// ============================================================================
// Purpose: Eliminate launch overhead for fixed-shape kernels
// Usage: Capture once, replay many times with zero CPU overhead
// ============================================================================

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace cudax {

// ============================================================================
// CUDA Graph Wrapper
// ============================================================================

class GraphExec {
public:
    GraphExec() = default;
    
    ~GraphExec() {
        if (graph_) cudaGraphDestroy(graph_);
        if (exec_) cudaGraphExecDestroy(exec_);
    }
    
    // No copy
    GraphExec(const GraphExec&) = delete;
    GraphExec& operator=(const GraphExec&) = delete;
    
    // Capture graph from stream
    template<typename Callable>
    void capture(cudaStream_t stream, Callable&& fn) {
        // Begin graph capture
        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaStreamBeginCapture failed: ") + cudaGetErrorString(err)
            );
        }
        
        // Execute user code (launches kernels into stream)
        try {
            fn();
        } catch (...) {
            cudaStreamEndCapture(stream, &graph_);
            throw;
        }
        
        // End capture
        err = cudaStreamEndCapture(stream, &graph_);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaStreamEndCapture failed: ") + cudaGetErrorString(err)
            );
        }
        
        // Instantiate executable graph
        err = cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGraphInstantiate failed: ") + cudaGetErrorString(err)
            );
        }
        
        captured_ = true;
    }
    
    // Replay captured graph
    void replay(cudaStream_t stream) {
        if (!captured_) {
            throw std::runtime_error("Graph not captured yet");
        }
        
        cudaError_t err = cudaGraphLaunch(exec_, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGraphLaunch failed: ") + cudaGetErrorString(err)
            );
        }
    }
    
    bool is_captured() const { return captured_; }
    
private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    bool captured_ = false;
};

// ============================================================================
// Helper: Capture + Replay Pattern
// ============================================================================

template<typename CaptureFunc, typename ValidateFunc>
void capture_and_verify(
    cudaStream_t stream,
    CaptureFunc&& capture_fn,
    ValidateFunc&& validate_fn
) {
    GraphExec graph;
    
    // Capture
    graph.capture(stream, std::forward<CaptureFunc>(capture_fn));
    
    // Warm-up replay
    for (int i = 0; i < 3; i++) {
        graph.replay(stream);
    }
    cudaStreamSynchronize(stream);
    
    // Validate output matches non-captured version
    validate_fn();
}

} // namespace cudax
