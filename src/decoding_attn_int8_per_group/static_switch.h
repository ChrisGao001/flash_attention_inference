// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:11:25 on Wed, Nov 29, 2023
//
// Description: static switch

#pragma once

#define BOOL_SWITCH(COND, CONST_NAME, ...)            \
    [&] {                                             \
        if (COND) {                                   \
            constexpr static bool CONST_NAME = true;  \
            return __VA_ARGS__();                     \
        } else {                                      \
            constexpr static bool CONST_NAME = false; \
            return __VA_ARGS__();                     \
        }                                             \
    }()

#define DECODING_INT8PG_FWD_HEADDIM_SWITCH(HEADDIM, ...) \
    [&] {                                                \
        if (HEADDIM == 64) {                             \
            constexpr static size_t HeadDim = 64;        \
            return __VA_ARGS__();                        \
        } else if (HEADDIM == 128) {                     \
            constexpr static size_t HeadDim = 128;       \
            return __VA_ARGS__();                        \
        } else if (HEADDIM == 256) {                     \
            constexpr static size_t HeadDim = 256;       \
            return __VA_ARGS__();                        \
        }                                                \
    }()
