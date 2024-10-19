#include "ggml-mm.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-athread/tensors.h"
#include <athread.h>
#include <cmath>
#include <cstddef>
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
    const struct ggml_tensor * src1 = dst->src[1];

    assert(ggml_is_contiguous(dst));
    assert(ggml_are_same_shape(src0, dst));

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);


    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    for (int i1 = 0; i1 < nr; i1++) {
        const float slope = 1.0;

        float * sp = (float *)((char *) src0->data + i1*src0->nb[1]);
        float * dp = (float *)((char *)  dst->data +  i1*dst->nb[1]);

        // broadcast the mask across rows
        ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((char *) src1->data) + (i1%ne01)*ne00 : NULL;
        float       * mp_f32 = src1 ? (float       *)((char *) src1->data) + (i1%ne01)*ne00 : NULL;

        ggml_vec_cpy_f32  (nc, wp, sp);
        ggml_vec_scale_f32(nc, wp, scale);
        if (mp_f32) {
            if (use_f16) {
                for (int i = 0; i < nc; ++i) {
                    wp[i] += slope*GGML_FP16_TO_FP32(mp_f16[i]);
                }
            } else {
                for (int i = 0; i < nc; ++i) {
                    wp[i] += slope*mp_f32[i];
                }
            }
        }

        float max = -INFINITY;
        ggml_vec_max_f32(nc, &max, wp);

        float sum = ggml_vec_soft_max_f32(nc, dp, wp, max);
        assert(sum > 0.0);

        sum = 1.0/sum;
        ggml_vec_scale_f32(nc, dp, sum);

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
           (op->op == GGML_OP_SOFT_MAX && ggml_backend_mm_use_softmax(op));

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
