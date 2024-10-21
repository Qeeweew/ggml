#include "ggml-mm.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-athread/tensors.h"
#include <athread.h>
#include <simd.h>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

static inline void to_athread_tensor(struct athread_tensor* dst, const struct ggml_tensor * src) {
    dst->data = src->data;
    for (int i = 0;i < GGML_MAX_DIMS;i++) {
        dst->nb[i] = src->nb[i];
        dst->ne[i] = src->ne[i];
    } 
}

static void ggml_set_op_params(struct ggml_tensor * tensor, const void * params, size_t params_size) {
    GGML_ASSERT(tensor != NULL); // silence -Warray-bounds warnings
    assert(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}


struct ggml_backend_mm_context {
    int64_t op_time[100];
    float* work_data;
};

// helper function to determine if it is better to use BLAS or not
// for large matrices, BLAS is faster
static bool ggml_backend_mm_use_mm(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    // TODO: find the optimal values for these
    if (nb0 == ggml_type_size(dst->type) &&
        nb00 == ggml_type_size(src0->type) &&
        nb10 == ggml_type_size(src1->type) &&
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)  && src1->type == GGML_TYPE_F32) {
        return true;
    }
    return false;
}

static bool ggml_backend_mm_use_softmax(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    if (scale == 1.0f && max_bias == 0.0f && src1 == nullptr) {
        return true;
    }
    return false;
}

static bool ggml_backend_mm_use_dup(const struct ggml_tensor * dst) {
    if (!ggml_is_contiguous(dst)) return false;
    if (dst->type != dst->src[0]->type) return false;
    return true;
}


extern "C" {
    void slave_mul_mat_fp32(void *);
    void slave_mul_mat_fp16(void *);
    void slave_cont_32(void *);
    void slave_softmax_f32(void *);
}


static inline void run_athread_func(void (*func)(void *), void * args) {
    __real_athread_spawn64_cgmask(1<<0, (void *) func, args);
    athread_join64_cgmask(1<<0);
}

static void ggml_backend_mm_mul_mat(ggml_backend_mm_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    mul_mat_args args;
    to_athread_tensor(&args.src0, src0);
    to_athread_tensor(&args.src1, src1);
    to_athread_tensor(&args.dst, dst);

    if (src0->type == GGML_TYPE_F32) {
        run_athread_func(slave_mul_mat_fp32, &args);
        return;
    } else {
        run_athread_func(slave_mul_mat_fp16, &args);
    }
}

/*
static inline void transpose8x8(int *dst, int ld0, int* src, int ld1) {
    static int mat[8][8] __attribute__((aligned(32)));
    intv8 r0,r1,r2,r3,r4,r5,r6,r7,r8;
    simd_load(r0, src + ld1 * 0);
    simd_load(r1, src + ld1 * 1);
    simd_load(r2, src + ld1 * 2);
    simd_load(r3, src + ld1 * 3);
    simd_load(r4, src + ld1 * 4);
    simd_load(r5, src + ld1 * 5);
    simd_load(r6, src + ld1 * 6);
    simd_load(r7, src + ld1 * 7);

    simd_store(r0, mat[0]);
    simd_store(r1, mat[1]);
    simd_store(r2, mat[2]);
    simd_store(r3, mat[3]);
    simd_store(r4, mat[4]);
    simd_store(r5, mat[5]);
    simd_store(r6, mat[6]);
    simd_store(r7, mat[7]);
    for (int i = 0;i < 8;i++) {
        for (int j = 0;j < i;j++) {
            int tmp = mat[i][j];
            mat[i][j] = mat[j][i];
            mat[j][i] = tmp; 
        }
    }
    simd_load(r0, mat[0]);
    simd_load(r1, mat[1]);
    simd_load(r2, mat[2]);
    simd_load(r3, mat[3]);
    simd_load(r4, mat[4]);
    simd_load(r5, mat[5]);
    simd_load(r6, mat[6]);
    simd_load(r7, mat[7]);
    simd_store(r0, dst + 0 * ld0);
    simd_store(r1, dst + 1 * ld0);
    simd_store(r2, dst + 2 * ld0);
    simd_store(r3, dst + 3 * ld0);
    simd_store(r4, dst + 4 * ld0);
    simd_store(r5, dst + 5 * ld0);
    simd_store(r6, dst + 6 * ld0);
    simd_store(r7, dst + 7 * ld0);
}
*/

// ggml_compute_forward_soft_max

static void ggml_backend_mm_soft_max(ggml_backend_mm_context * ctx, struct ggml_tensor * dst) {

    unary_args args;
    to_athread_tensor(&args.dst, dst);
    to_athread_tensor(&args.src0, dst->src[0]);
    run_athread_func(slave_softmax_f32, &args);
}


static void ggml_backend_mm_dup(ggml_backend_mm_context * ctx, struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    const size_t type_size = ggml_type_size(src0->type);

    int i0, i1, i2, i3;
    char* dst_ptr = (char *) dst->data;
    char* src_ptr = (char *) src0->data;

    if (type_size == 2) {
        for (i3 = 0; i3 < ne03; i3++) {
            for (i2 = 0; i2 < ne02; i2++) {
                for (i1 = 0; i1 < ne01; i1++) {
                    for (i0 = 0; i0 < ne00; i0++) {
                        *(uint16_t*) (dst_ptr + i0 * 2 + i1 * ne00 * 2 + i2 * ne00 * ne01 * 2 + i3 * ne00 * ne01 * ne02 * 2) = 
                        *(uint16_t*) (src_ptr + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03);
                    }
                }
            }
        }
    } else if (type_size == 4) {
        if (ne00 * ne01 * ne02 * ne03 <= 1024) {
            for (i3 = 0; i3 < ne03; i3++) {
                for (i2 = 0; i2 < ne02; i2++) {
                    for (i1 = 0; i1 < ne01; i1++) {
                        for (i0 = 0; i0 < ne00; i0++) {
                            *(uint32_t*) (dst_ptr + i0 * 4 + i1 * ne00 * 4 + i2 * ne00 * ne01 * 4 + i3 * ne00 * ne01 * ne02 * 4) = 
                            *(uint32_t*) (src_ptr + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03);
                        }
                    }
                }
            }
            return;
        }
        unary_args args;
        to_athread_tensor(&args.dst, dst);
        to_athread_tensor(&args.src0, src0);
        run_athread_func(slave_cont_32, &args);
    } else {
        assert(false);
    }
}

// backend interface

static const char * ggml_backend_mm_name(ggml_backend_t backend) {
    return "mm";

    GGML_UNUSED(backend);
}

static void ggml_backend_mm_free(ggml_backend_t backend) {
    athread_leave64_arg();
    ggml_backend_mm_context * ctx = (ggml_backend_mm_context *)backend->context;

    printf("\nOP TIME USED:\n");
    int64_t * op_time = ctx->op_time;
    for (int i = 0;i < 100;i++) {
        if (op_time[i] != 0) {
            printf("%s : %ld\n", ggml_op_name(ggml_op(i)), op_time[i]); 
        }
    }
    printf("\n");


    delete ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_mm_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_mm_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_mm_context * ctx = (ggml_backend_mm_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        int64_t start = ggml_time_us();

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_mm_mul_mat(ctx, node);
                break;
            case GGML_OP_SOFT_MAX:
                ggml_backend_mm_soft_max(ctx, node);
                break;
            case GGML_OP_DUP:
                ggml_backend_mm_dup(ctx, node);
                break;
            case GGML_OP_CONT:
                ggml_backend_mm_dup(ctx, node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
        int64_t end = ggml_time_us();
        ctx->op_time[node->op] += end - start;
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static bool ggml_backend_mm_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    return (op->op == GGML_OP_MUL_MAT  && ggml_backend_mm_use_mm(op)) ||
           (op->op == GGML_OP_SOFT_MAX && ggml_backend_mm_use_softmax(op)) ||
           (op->op == GGML_OP_DUP && ggml_backend_mm_use_dup(op)) ||
           (op->op == GGML_OP_CONT && ggml_backend_mm_use_dup(op));

    GGML_UNUSED(backend);
}

static bool ggml_backend_mm_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(backend);
}

static struct ggml_backend_i mm_backend_i = {
    /* .get_name                = */ ggml_backend_mm_name,
    /* .free                    = */ ggml_backend_mm_free,
    /* .get_default_buffer_type = */ ggml_backend_mm_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_mm_graph_compute,
    /* .supports_op             = */ ggml_backend_mm_supports_op,
    /* .supports_buft           = */ ggml_backend_mm_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_mm_guid(void) {
    static ggml_guid guid = { 0x32, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}

ggml_backend_t ggml_backend_mm_init(void) {
    athread_res_show();
    athread_enter64_arg();

    // int tmp[8][8];

    // for (int i = 0;i < 8;i++) {
    //     for (int j = 0;j < 8;j++) {
    //         tmp[i][j] = i * 8 + j;
    //     }
    // }
    // for (int i = 0;i < 8;i++) {
    //     for (int j = 0;j < 8;j++) {
    //         printf("%d ", tmp[i][j]);
    //     }
    //     printf("\n");
    // }
    // transpose8x8((int *) tmp, 8, (int *) tmp, 8);
    // for (int i = 0;i < 8;i++) {
    //     for (int j = 0;j < 8;j++) {
    //         printf("%d ", tmp[i][j]);
    //     }
    //     printf("\n");
    // }


    ggml_backend_mm_context * ctx = new ggml_backend_mm_context;
    memset(ctx->op_time,0, sizeof(ctx->op_time));
    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_mm_guid(),
        /* .interface = */ mm_backend_i,
        /* .device    = */ nullptr,
        /* .context   = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_mm(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_mm_guid());
}
