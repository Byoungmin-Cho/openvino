// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/include_all.cl"

#define SIMD 16
#define FSV 16
#define unroll_for  __attribute__((opencl_unroll_hint(READ_OFFSET))) for

#define CEIL_DIV(a, b) (((a) + (b) - 1)/(b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))

#if !defined REDUCE_BATCH
    #define REDUCE_BATCH 0
#endif
#if !defined REDUCE_FEATURE
    #define REDUCE_FEATURE 0
#endif
#if !defined REDUCE_Y
    #define REDUCE_Y 0
#endif
#if !defined REDUCE_X
    #define REDUCE_X 0
#endif

#define INPUT_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_OFFSET)

#define ACCUMULATOR_VEC MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, READ_OFFSET)
#define TO_ACCUMULATOR_VEC CAT(convert_, ACCUMULATOR_VEC)
#define FINAL_ACCUMULATOR_VEC MAKE_VECTOR_TYPE(FINAL_ACCUMULATOR_TYPE, READ_OFFSET)

#define ACTIVATION_VEC MAKE_VECTOR_TYPE(ACTIVATION_TYPE, READ_OFFSET)
#define TO_ACTIVATION_VEC CAT(convert_, ACTIVATION_VEC)

#define OUTPUT_VEC MAKE_VECTOR_TYPE(OUTPUT_TYPE, READ_OFFSET)
#define TO_OUTPUT_VEC CAT(convert_, OUTPUT_VEC)

#define REDUCE_BFY_BY_FY_Y          REDUCE_BATCH && REDUCE_FEATURE && REDUCE_Y && !REDUCE_X || REDUCE_BATCH && REDUCE_Y && !REDUCE_FEATURE && !REDUCE_X || \
                                    REDUCE_FEATURE && REDUCE_Y && !REDUCE_BATCH && !REDUCE_X|| REDUCE_Y && !REDUCE_BATCH && !REDUCE_FEATURE && !REDUCE_X

#define REDUCE_F                    REDUCE_FEATURE && !REDUCE_BATCH && !REDUCE_Y && !REDUCE_X

#define NEED_SUB_GROUP_REDUCE       REDUCE_FEATURE

#define INIT_VAL ACCUMULATOR_VAL_ZERO
#define INPUT_INIT_VAL INPUT0_VAL_ZERO

inline ACCUMULATOR_TYPE FUNC(apply_reduce)(ACCUMULATOR_TYPE acc, ACCUMULATOR_TYPE input) {
    acc += input;
    return acc;
}

inline ACCUMULATOR_TYPE FUNC(sub_group_reduce)(ACCUMULATOR_TYPE acc) {
    // #if NEED_SUB_GROUP_REDUCE
    //     acc = sub_group_reduce_add(acc);
    // #endif

    return acc;
}

inline FINAL_ACCUMULATOR_TYPE FUNC(final_reduce)(FINAL_ACCUMULATOR_TYPE acc) {
    acc /= DIVIDER;
    return acc;
}

inline uint FUNC(calc_linear_offset)(uint b, uint f, uint y, uint x) {
    uint index = b * COMMON_OUTPUT_SIZE_X * COMMON_OUTPUT_SIZE_Y * COMMON_OUTPUT_FEATURE_NUM +
                 f * COMMON_OUTPUT_SIZE_X * COMMON_OUTPUT_SIZE_Y +
                 y * COMMON_OUTPUT_SIZE_X +
                 x;

    return index;
}

__attribute__((intel_reqd_sub_group_size(SIMD)))
KERNEL(reduce_fsv16)(
    const __global INPUT0_TYPE* data,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    __local ACCUMULATOR_TYPE lg_storage[BLOCK_Y_NUM];

    const uint lid  = (uint)get_local_id(1);

    const uint bf   = (uint)get_global_id(2) * SIMD;
    const uint b    = bf / ALIGN(COMMON_OUTPUT_FEATURE_NUM, SIMD);
    const uint f    = bf % ALIGN(COMMON_OUTPUT_FEATURE_NUM, SIMD);

    const uint out_idx = OUTPUT_GET_INDEX(b, f, 0, 0);

    const uint linear_idx = FUNC_CALL(calc_linear_offset)(b, f, 0, 0);

    if (linear_idx >= COMPUTATIONAL_OPERATIONS_NUMBER)
        return;

    const uint input_x_pitch = FSV;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_batch_pitch = input_fs_pitch * ((INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM + FSV - 1) / FSV);
    const uint padding_pitch = INPUT0_GET_INDEX(0, 0, 0, 0);

    const uint output_x_pitch = FSV;
    const uint output_y_pitch = FSV * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);

    const uint batch_out = BATCH_NUM_IDX_COMP(linear_idx);
    const uint batch_max_val = batch_out + 1;

    const uint feature_out = FEATURE_NUM_IDX_COMP(linear_idx);
    const uint feature_max_val = feature_out + 1;

    const uint y_out = (uint)get_local_id(1) * BLOCK_Y_SIZE;
    const uint y_max_val = y_out + BLOCK_Y_SIZE;

    const uint x_out = 0;
    const uint x_max_val = INPUT0_SIZE_X / READ_OFFSET;
    const uint x_leftover_start = x_max_val * READ_OFFSET;
    const uint x_leftover_end = INPUT0_SIZE_X;

    uint offset = batch_out * input_batch_pitch + ((feature_out + FSV - 1) / FSV) * input_fs_pitch + y_out * input_y_pitch + x_out * input_x_pitch + padding_pitch;

    ACCUMULATOR_TYPE acc = INIT_VAL;
    for (uint bi = batch_out; bi < batch_max_val; ++bi) {
        for (uint fi = feature_out; fi < feature_max_val; fi += FSV) {
            for (uint yi = y_out; yi < y_max_val; ++yi) {
                for (uint xi = x_out; xi < x_max_val; ++xi) {
                    INPUT_VEC input = (INPUT_VEC)(INPUT_INIT_VAL);
                    input = BLOCK_READ(data, offset); // DT_INPUT_BLOCK_READ4(ptr,offset)
                    unroll_for (int i = 0; i < READ_OFFSET; ++i)
                        acc += input[i]; // acc = FUNC_CALL(apply_reduce)(acc, input[i]);
                    offset += input_x_pitch * READ_OFFSET;
                }
                #if INPUT0_SIZE_X % READ_OFFSET != 0
                    for (uint xi = x_leftover_start; xi < x_leftover_end; ++xi) {
                        INPUT0_TYPE leftovers = INIT_VAL;
                        leftovers = DT_INPUT_BLOCK_READ(data, offset); // BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
                        acc += leftovers; // acc = FUNC_CALL(apply_reduce)(acc, leftovers);
                        offset += input_x_pitch;
                    }
                #endif
                offset += input_y_pitch - INPUT0_SIZE_X * input_x_pitch;
            }
            offset += input_fs_pitch - ((y_max_val - y_out) * input_y_pitch);
        }
        offset += input_batch_pitch - ((((feature_max_val - feature_out) + FSV - 1) / FSV) * input_fs_pitch);
    }

    lg_storage[lid] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid != 0)
        return;

    acc = INIT_VAL;
    unroll_for (uint i = 0; i < BLOCK_Y_NUM; i++) {
        acc += lg_storage[i];
    }

    FINAL_ACCUMULATOR_TYPE final_acc;
    acc = FUNC_CALL(sub_group_reduce)(acc);
    final_acc = FUNC_CALL(final_reduce)(TO_FINAL_ACCUMULATOR_TYPE(acc));

    OUTPUT_TYPE final_result;
    ACTIVATION_TYPE reduce_result = TO_ACTIVATION_TYPE(final_acc);
    #if HAS_FUSED_OPS
        FUSED_OPS_SCALAR;
        final_result = FUSED_OPS_RESULT_SCALAR;
    #else
        final_result = TO_OUTPUT_TYPE(ACTIVATION(reduce_result, ACTIVATION_PARAMS));
    #endif

    // #if (REDUCE_FEATURE && REDUCE_X || REDUCE_BATCH && REDUCE_X) && !KEEP_DIMS
    //     output[out_idx] = final_result;
    // #elif REDUCE_BATCH && REDUCE_Y && REDUCE_X || REDUCE_BATCH && REDUCE_X || REDUCE_Y && REDUCE_X || REDUCE_X && !REDUCE_FEATURE
        DT_OUTPUT_BLOCK_WRITE(output + out_idx, 0, final_result);
    // #else
    //     if (get_sub_group_local_id() == 0)
    //         output[out_idx] = final_result;
    // #endif
// #else // REDUCE_X
//     ACCUMULATOR_VEC acc = (ACCUMULATOR_VEC)(INIT_VAL);
//     for (uint bi = batch_out; bi < batch_max_val; ++bi) {
//         for (uint fi = feature_out; fi < feature_max_val; fi += FSV) {
//             for (uint yi = y_out; yi < y_max_val; ++yi) {
//                 for (uint xi = x_out; xi < x_max_val; ++xi) {
//                     INPUT_VEC input = (INPUT_VEC)(INPUT_INIT_VAL);
//                     input = BLOCK_READ(data, offset);
//                     unroll_for (int i = 0; i < READ_OFFSET; ++i)
//                         acc[i] = FUNC_CALL(apply_reduce)(acc[i], input[i]);
//                     offset += input_x_pitch;
//                 }
//                 offset += input_y_pitch - (x_max_val - x_out) * input_x_pitch;
//             }
//             offset += input_fs_pitch - ((y_max_val - y_out) * input_y_pitch);
//         }
//         offset += input_batch_pitch - ((((feature_max_val - feature_out) + FSV - 1) / FSV) * input_fs_pitch);
//     }

//     FINAL_ACCUMULATOR_VEC final_acc;
//     unroll_for (uint i = 0; i < READ_OFFSET; ++i) {
//         acc[i] = FUNC_CALL(sub_group_reduce)(acc[i]);
//         final_acc[i] = FUNC_CALL(final_reduce)(TO_FINAL_ACCUMULATOR_TYPE(acc[i]));
//     }

//     OUTPUT_VEC final_result;
//     ACTIVATION_VEC reduce_result = TO_ACTIVATION_VEC(final_acc);

//     #if HAS_FUSED_OPS
//         FUSED_OPS_VECTOR;
//         final_result = (OUTPUT_VEC)(FUSED_OPS_RESULT_VECTOR);
//     #else
//         final_result = TO_OUTPUT_VEC(ACTIVATION(reduce_result, ACTIVATION_PARAMS));
//     #endif

//     unroll_for (uint i = 0; i < READ_OFFSET; ++i) {
//         if(COMMON_OUTPUT_SIZE_X % READ_OFFSET == 0 || x + i < COMMON_OUTPUT_SIZE_X) {
//             #if REDUCE_BATCH && REDUCE_FEATURE && REDUCE_Y && !REDUCE_X && !KEEP_DIMS
//                 output[out_idx + output_x_pitch * i] = final_result[i];
//             #elif REDUCE_FEATURE && REDUCE_Y && !KEEP_DIMS
//                 if (get_sub_group_local_id() == 0)
//                     output[out_idx + i] = final_result[i];
//             #elif REDUCE_BATCH && REDUCE_Y && !KEEP_DIMS
//                     output[out_idx + i] = final_result[i];
//             #elif REDUCE_BATCH && REDUCE_Y && REDUCE_X && !KEEP_DIMS
//                     output[out_idx + get_sub_group_local_id() + output_y_pitch * i] = final_result[i];
//             #elif REDUCE_BFY_BY_FY_Y
//                     output[out_idx + get_sub_group_local_id() + output_x_pitch * i] = final_result[i];
//             #elif REDUCE_BATCH && REDUCE_FEATURE && !KEEP_DIMS
//                 if (get_sub_group_local_id() == 0)
//                     output[out_idx + i] = final_result[i];
//             #elif REDUCE_BATCH && !KEEP_DIMS
//                     output[out_idx + output_y_pitch * i] = final_result[i];
//             #elif REDUCE_BATCH && !REDUCE_FEATURE
//                     DT_OUTPUT_BLOCK_WRITE(output + out_idx + output_x_pitch * i, 0, final_result[i]);
//             #elif REDUCE_BATCH && REDUCE_FEATURE
//                     if (get_sub_group_local_id() == 0)
//                         output[out_idx + output_x_pitch * i] = final_result[i];
//             #elif REDUCE_F && !KEEP_DIMS
//                     if (get_sub_group_local_id() == 0)
//                         output[out_idx + output_y_pitch * i] = final_result[i];
//             #elif REDUCE_F
//                     if (get_sub_group_local_id() == 0)
//                         output[out_idx + output_x_pitch * i] = final_result[i];
//             #endif
//         }
//     }
// #endif
}

#undef SIMD
#undef FSV
#undef unroll_for
#undef BLOCK_READ
#undef READ_OFFSET
#undef INPUT_VEC
#undef ACCUMULATOR_VEC
#undef TO_ACCUMULATOR_VEC
#undef FINAL_ACCUMULATOR_VEC
#undef ACTIVATION_VEC
#undef TO_ACTIVATION_VEC
#undef OUTPUT_VEC
#undef TO_OUTPUT_VEC
#undef REDUCE_BFY_BY_FY_Y
#undef REDUCE_F
#undef NEED_SUB_GROUP_REDUCE
#undef INIT_VAL
#undef INPUT_INIT_VAL
#undef REDUCE_BATCH
#undef REDUCE_FEATURE
#undef REDUCE_Y
#undef REDUCE_X
