// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:33:18 on Tue, Nov 07, 2023
//
// Description: decoding int8ph fwd hdim256

#include "decoding_attn_int8_per_head/decoding_int8ph_fwd_launch_template.h"

template <>
void run_quantization_int8ph_<256>(const DecodingInt8PHParams &params) {
    quantization_int8ph<256, 512, 16>(params);
}

template <>
void run_mha_decoding_int8ph_fwd_<256>(const DecodingInt8PHParams &params) {
    mha_decoding_int8ph_fwd<256, 256, 16>(params);
}
