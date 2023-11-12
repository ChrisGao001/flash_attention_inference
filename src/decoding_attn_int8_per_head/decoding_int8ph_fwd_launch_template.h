// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:33:18 on Tue, Nov 07, 2023
//
// Description: decoding int8ph fwd launch template

#pragma once

#include "decoding_attn_int8_per_head/decoding_int8ph_fwd_kernel.h"
#include "decoding_attn_int8_per_head/static_switch.h"

template <size_t HeadDim, size_t ThreadsPerBlock, size_t ThreadsPerGroup>
void quantization_int8ph(const DecodingInt8PHParams &params) {
    dim3 block(ThreadsPerBlock);
    dim3 grid(params.b, params.seqlen_k);

    quantization_int8ph_kernel<QuantizationInt8KernelTraits<HeadDim, ThreadsPerBlock, ThreadsPerGroup>>
        <<<grid, block, 0, params.stream>>>(params);
    FAI_CHECK_CUDART_ERROR(cudaPeekAtLastError());
}

template <size_t HeadDim, size_t ThreadsPerBlock, size_t ThreadsPerGroup>
void mha_decoding_int8ph_fwd(const DecodingInt8PHParams &params) {
    constexpr size_t warp_size = 32;
    constexpr size_t static_smem_size = ThreadsPerBlock / warp_size * sizeof(float);
    const size_t dynamic_smem_size = std::max(params.seqlen_k * sizeof(float), params.d * sizeof(float));
    FAI_CHECK_GT(params.props->sharedMemPerBlock, static_smem_size + dynamic_smem_size);

    dim3 block(ThreadsPerBlock);
    dim3 grid(params.b, params.h);

    BOOL_SWITCH(params.is_alibi, IsAlibi, [&] {
        mha_decoding_int8ph_fwd_kernel<DecodingInt8KernelTraits<HeadDim, ThreadsPerBlock, ThreadsPerGroup>, IsAlibi>
            <<<grid, block, dynamic_smem_size, params.stream>>>(params);
        FAI_CHECK_CUDART_ERROR(cudaPeekAtLastError());
    });
}
