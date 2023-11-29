// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:11:25 on Wed, Nov 29, 2023
//
// Description: decoding int8pg fwd launch template

#pragma once

#include "decoding_attn_int8_per_group/decoding_int8pg_fwd_kernel.h"
#include "decoding_attn_int8_per_group/static_switch.h"

template <size_t HeadDim, size_t ThreadsPerBlock, size_t QuantizationGroupSize>
void quantization_int8pg(const DecodingInt8PGParams &params) {
    dim3 block(ThreadsPerBlock);
    dim3 grid(params.b, params.seqlen_k);

    quantization_int8pg_kernel<QuantizationInt8PGKernelTraits<HeadDim, ThreadsPerBlock, QuantizationGroupSize>>
        <<<grid, block, 0, params.stream>>>(params);
    FAI_CHECK_CUDART_ERROR(cudaPeekAtLastError());
}

template <size_t HeadDim, size_t ThreadsPerBlock, size_t ThreadsPerGroup, size_t QuantizationGroupSize>
void mha_decoding_int8pg_fwd(const DecodingInt8PGParams &params) {
    constexpr size_t warp_size = 32;
    constexpr size_t static_smem_size = ThreadsPerBlock / warp_size * sizeof(float);
    const size_t dynamic_smem_size = std::max(params.seqlen_k * sizeof(float), params.d * sizeof(float));
    FAI_CHECK_GT(params.props->sharedMemPerBlock, static_smem_size + dynamic_smem_size);

    dim3 block(ThreadsPerBlock);
    dim3 grid(params.b, params.h);

    BOOL_SWITCH(params.is_alibi, IsAlibi, [&] {
        mha_decoding_int8pg_fwd_kernel<
            DecodingInt8PGKernelTraits<HeadDim, ThreadsPerBlock, ThreadsPerGroup, QuantizationGroupSize>, IsAlibi>
            <<<grid, block, dynamic_smem_size, params.stream>>>(params);
        FAI_CHECK_CUDART_ERROR(cudaPeekAtLastError());
    });
}
