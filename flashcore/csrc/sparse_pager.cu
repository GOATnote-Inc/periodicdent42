// SPDX-License-Identifier: Apache-2.0
// FlashCore integration of SGLang's radix_sparse CSR paging
// Attribution: SGLang team (sparse paging concept), FlashCore (integration)
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__
void build_csr_and_stage_pages(
    const int32_t* __restrict__ token_to_page,
    const int32_t* __restrict__ seq_starts,
    const int32_t* __restrict__ seq_ends,
    const uint8_t* __restrict__ page_resident_bitmap,
    int32_t* __restrict__ row_offsets,
    int32_t* __restrict__ cols,
    int32_t* __restrict__ token_counts,
    int32_t* __restrict__ nnz_out,
    int32_t* __restrict__ staging_page_ids,
    int32_t* __restrict__ staging_size,
    const int32_t PAGE_TOKENS,
    const int32_t num_pages
){
    const int b = blockIdx.x;
    if (b >= gridDim.x) return;

    const int32_t start = seq_starts[b];
    const int32_t end   = seq_ends[b];

    int32_t write_ptr = 0;
    int32_t last_page = -1;
    int32_t run_len   = 0;

    int32_t row_base = atomicAdd(nnz_out, 0);
    row_offsets[b] = row_base;

    for (int32_t t = start; t < end; ++t) {
        const int32_t page = token_to_page[t];
        if (page != last_page) {
            if (run_len > 0) {
                cols[row_base + write_ptr] = last_page;
                token_counts[row_base + write_ptr] = run_len;
                write_ptr++;
            }
            last_page = page;
            run_len = 1;
            if (page_resident_bitmap && page_resident_bitmap[page] == 0) {
                int sidx = atomicAdd(staging_size, 1);
                staging_page_ids[sidx] = page;
            }
        } else {
            run_len++;
            if (run_len == PAGE_TOKENS) {
                cols[row_base + write_ptr] = last_page;
                token_counts[row_base + write_ptr] = run_len;
                write_ptr++;
                last_page = -1;
                run_len = 0;
            }
        }
    }
    if (run_len > 0) {
        cols[row_base + write_ptr] = last_page;
        token_counts[row_base + write_ptr] = run_len;
        write_ptr++;
    }
    atomicAdd(nnz_out, write_ptr);
    row_offsets[b+1] = row_base + write_ptr;
}

