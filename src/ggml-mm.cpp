#include "ggml-mm.h"
#include "ggml-athread/pt_math.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-athread/tensors.h"
#include <athread.h>
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
    float* work_data;
};

// helper function to determine if it is better to use BLAS or not
// for large matrices, BLAS is faster
static bool ggml_backend_mm_use_mm(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if (!ggml_is_permuted(src0) &&
        !ggml_is_permuted(src1) &&
        !ggml_is_permuted(dst) &&
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)  && src1->type == GGML_TYPE_F32 && ne10 % 32 == 0) {
        /*printf("BLAS: %d %d %d %d %d\n", ne0, ne1, ne10, ne00, ne01);*/
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

// ggml_compute_forward_soft_max

static void ggml_backend_mm_soft_max(ggml_backend_mm_context * ctx, struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    for (int i1 = 0; i1 < nr; i1++) {
        float * sp = (float *)((char *) src0->data + i1*src0->nb[1]);
        float * dp = (float *)((char *)  dst->data +  i1*dst->nb[1]);

        float max = -FLT_MAX / 2;
        for (int j = 0;j < nc;j++) {
            max = fmax(max, sp[j]);
        }
        float sum = 0.0f;
        for (int j = 0;j < nc;j++) {
            dp[j] = sp[j] - max < -20.0f ? 0.0f : expf(sp[j] - max);
            sum += dp[j];
        }
        assert(sum > 0.0);
        sum = 1.0/sum;
        for (int j = 0;j < nc;j++) {
            dp[j] = dp[j] * sum;
        }
    }
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
        if (nb00 == type_size) {
            for (i3 = 0; i3 < ne03; i3++) {
                for (i2 = 0; i2 < ne02; i2++) {
                    for (i1 = 0; i1 < ne01; i1++) {
#pragma GCC unroll 8
                        for (i0 = 0; i0 < ne00; i0++) {
                            *(uint32_t*) (dst_ptr + i0 * 4 + i1 * ne00 * 4 + i2 * ne00 * ne01 * 4 + i3 * ne00 * ne01 * ne02 * 4) = 
                            *(uint32_t*) (src_ptr + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03);
                        }
                    }
                }
            }
        } else if (nb01 == type_size) {
            for (i3 = 0; i3 < ne03; i3++) {
                for (i2 = 0; i2 < ne02; i2++) {
                    for (i1 = 0; i1 + 8 <= ne01; i1+=8) {
                        for (i0 = 0; i0 < ne00; i0++) {
#pragma GCC unroll 8 
                            for (int p = 0; p < 8;p++) {
                                *(uint32_t*) (dst_ptr + i0 * 4 + (i1 + p) * ne00 * 4 + i2 * ne00 * ne01 * 4 + i3 * ne00 * ne01 * ne02 * 4) = 
                                *(uint32_t*) (src_ptr + i0 * nb00 + (i1 + p) * nb01 + i2 * nb02 + i3 * nb03);
                            }
                        }
                    }
                    for (; i1 < ne01; i1++) {
                        for (i0 = 0; i0 < ne00; i0++) {
                            *(uint32_t*) (dst_ptr + i0 * 4 + i1 * ne00 * 4 + i2 * ne00 * ne01 * 4 + i3 * ne00 * ne01 * ne02 * 4) = 
                            *(uint32_t*) (src_ptr + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03);
                        }
                    }
                }
            }
        } else if (nb02 == type_size) {
            for (i3 = 0; i3 < ne03; i3++) {
                for (i1 = 0; i1 < ne01; i1++) {
                    for (i2 = 0; i2 <= ne02; i2+=8) {
                        for (i0 = 0; i0 < ne00; i0++) {
#pragma GCC unroll 8 
                            for (int p = 0; p < 8;p++) {
                                *(uint32_t*) (dst_ptr + i0 * 4 + i1 * ne00 * 4 + (i2 + p) * ne00 * ne01 * 4 + i3 * ne00 * ne01 * ne02 * 4) = 
                                *(uint32_t*) (src_ptr + i0 * nb00 + i1 * nb01 + (i2 + p) * nb02 + i3 * nb03);
                            }
                        }
                    }
                    for (; i2 < ne02; i2++) {
                        for (i0 = 0; i0 < ne00; i0++) {
                            *(uint32_t*) (dst_ptr + i0 * 4 + i1 * ne00 * 4 + i2 * ne00 * ne01 * 4 + i3 * ne00 * ne01 * ne02 * 4) = 
                            *(uint32_t*) (src_ptr + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03);
                        }
                    }
                }
            }
        } else {
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
        }

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

    ggml_backend_mm_context * ctx = new ggml_backend_mm_context;
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
