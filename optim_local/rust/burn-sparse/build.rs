// Build script to compile CUDA kernels and link with Rust

use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to link the CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cusparse");
    println!("cargo:rustc-link-lib=curand");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    
    // Compile CUDA kernels
    let cuda_files = vec![
        "../../src/sparse/bsr_kernel_64.cu",
        "../../src/sparse/cusparse_baseline.cu",
    ];
    
    cc::Build::new()
        .cuda(true)
        .flag("-O3")
        .flag("--expt-relaxed-constexpr")
        .flag("-arch=sm_90")  // H100
        .flag("--use_fast_math")
        .include("/usr/local/cuda/include")
        .include("/opt/cutlass/include")
        .include("../../src/sparse")
        .files(&cuda_files)
        .compile("sparse_kernels");
    
    // Tell cargo to rerun if CUDA files change
    for file in cuda_files {
        println!("cargo:rerun-if-changed={}", file);
    }
}
