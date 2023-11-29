// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:11:25 on Wed, Nov 29, 2023
//
// Description: decoding int8pg fwd hdim64

#include "decoding_attn_int8_per_group/decoding_int8pg_fwd_launch_template.h"

template <>
void run_quantization_int8pg_<64>(const DecodingInt8PGParams &params) {
    quantization_int8pg<64, 256, QUANTIZATION_GROUP_SIZE>(params);
}

template <>
void run_mha_decoding_int8pg_fwd_<64>(const DecodingInt8PGParams &params) {
    mha_decoding_int8pg_fwd<64, 256, 4, QUANTIZATION_GROUP_SIZE>(params);
}
