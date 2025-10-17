#pragma once

namespace cudadent {

enum class Impl {
    CUSTOM_V3,    // Custom FlashAttention (Phase 4)
    CUBLAS,       // cuBLAS for Q@K^T and P@V
    CUTENSOR,     // cuTENSOR for tensor operations
    WMMA,         // WMMA Tensor Cores (manual)
    HYBRID_QKT,   // cuBLAS Q@K^T + custom P@V
    HYBRID_PV     // Custom Q@K^T + cuBLAS P@V
};

inline const char* impl_to_string(Impl impl) {
    switch (impl) {
        case Impl::CUSTOM_V3:   return "custom_v3";
        case Impl::CUBLAS:      return "cublas";
        case Impl::CUTENSOR:    return "cutensor";
        case Impl::WMMA:        return "wmma";
        case Impl::HYBRID_QKT:  return "hybrid_qkt";
        case Impl::HYBRID_PV:   return "hybrid_pv";
        default:                return "unknown";
    }
}

inline Impl string_to_impl(const char* str) {
    if (!str) return Impl::CUSTOM_V3;
    
    std::string s(str);
    if (s == "cublas")      return Impl::CUBLAS;
    if (s == "cutensor")    return Impl::CUTENSOR;
    if (s == "wmma")        return Impl::WMMA;
    if (s == "hybrid_qkt")  return Impl::HYBRID_QKT;
    if (s == "hybrid_pv")   return Impl::HYBRID_PV;
    
    return Impl::CUSTOM_V3;  // Default
}

inline Impl get_impl_from_env() {
    const char* impl_str = std::getenv("IMPL");
    return string_to_impl(impl_str);
}

} // namespace cudadent

