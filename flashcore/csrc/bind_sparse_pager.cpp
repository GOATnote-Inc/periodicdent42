// SPDX-License-Identifier: Apache-2.0
// PyTorch bindings for FlashCore sparse pager
#include <torch/extension.h>

void build_csr_and_stage_pages(
    const int32_t* token_to_page,
    const int32_t* seq_starts,
    const int32_t* seq_ends,
    const uint8_t* page_resident_bitmap,
    int32_t* row_offsets,
    int32_t* cols,
    int32_t* token_counts,
    int32_t* nnz_out,
    int32_t* staging_page_ids,
    int32_t* staging_size,
    int32_t   PAGE_TOKENS,
    int32_t   num_pages);

torch::Tensor radix_sparse_build_layout(torch::Tensor token_to_page,
                                        torch::Tensor seq_starts,
                                        torch::Tensor seq_ends,
                                        torch::Tensor page_resident_bitmap,
                                        int page_tokens,
                                        int num_pages) {
    const int B = seq_starts.size(0);

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(token_to_page.device());

    auto row_offsets = torch::empty({B+1}, opts_i32);
    auto cols        = torch::empty({B * (128*1024 / page_tokens + 2)}, opts_i32);
    auto counts      = torch::empty_like(cols);
    auto nnz_out     = torch::zeros({1}, opts_i32);

    auto staging_ids   = torch::empty({B * 16}, opts_i32);
    auto staging_size  = torch::zeros({1}, opts_i32);

    dim3 grid(B);
    dim3 block(1);

    build_csr_and_stage_pages<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        token_to_page.data_ptr<int32_t>(),
        seq_starts.data_ptr<int32_t>(),
        seq_ends.data_ptr<int32_t>(),
        page_resident_bitmap.defined() ? page_resident_bitmap.data_ptr<uint8_t>() : nullptr,
        row_offsets.data_ptr<int32_t>(),
        cols.data_ptr<int32_t>(),
        counts.data_ptr<int32_t>(),
        nnz_out.data_ptr<int32_t>(),
        staging_ids.data_ptr<int32_t>(),
        staging_size.data_ptr<int32_t>(),
        page_tokens,
        num_pages
    );
    auto nnz = nnz_out.item<int32_t>();
    auto staged = staging_size.item<int32_t>();
    return torch::stack({row_offsets,
                         cols.index({torch::indexing::Slice(0, nnz)}),
                         counts.index({torch::indexing::Slice(0, nnz)}),
                         staging_ids.index({torch::indexing::Slice(0, staged)})});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_layout", &radix_sparse_build_layout, "Build CSR + staging list for radix_sparse FlashCore");
}

