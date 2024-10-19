#include "ggml-mm.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include <cmath>
#include <cstring>

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
        (src0->type == GGML_TYPE_F32)  && src1->type == GGML_TYPE_F32) {
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



static void ggml_backend_mm_mul_mat(ggml_backend_mm_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            const char *x = ((char *) src0->data + i02 * nb02 + i03 * nb03);
            const char *y = ((char *) src1->data + i12 * nb12 + i13 * nb13);
            char *d = ((char *) dst->data + i12 * nb2 + i13 * nb3);

            // Perform the matrix multiplication with initialization
            for (int64_t i = 0; i < ne1; i++) {
                const float *y_row = (const float *) (y + nb11 * i);
                for (int64_t j = 0; j < ne01; j++) {
                    const float *x_row = (const float *) (x + nb01 * j);
                    float sum = 0.0f;
                    for (int64_t k = 0; k < ne10; k++) {
                        sum += y_row[k] * x_row[k];
                    }
                    *((float *) (d + i * nb1 + j * nb0)) = sum;
                }
            }
        }
    }
}

static void ggml_backend_mm_soft_max(ggml_backend_mm_context * ctx, struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    for (int i1 = 0; i1 < nr; i1++) {
        float * sp = (float *)((char *) src0->data + i1*src0->nb[1]);
        float * dp = (float *)((char *)  dst->data +  i1*dst->nb[1]);

        float max = -INFINITY;
        for (int j = 0;j < nc;j++) {
            max = fmax(max, sp[j]);
        }
        float sum = 0.0f;
        for (int j = 0;j < nc;j++) {
            dp[j] = expf(sp[j] - max);
            sum += dp[j];
        }
        assert(sum > 0.0);
        sum = 1.0/sum;
        for (int j = 0;j < nc;j++) {
            dp[j] = dp[j] * sum;
        }
    }
}

// backend interface

static const char * ggml_backend_mm_name(ggml_backend_t backend) {
    return "mm";

    GGML_UNUSED(backend);
}

static void ggml_backend_mm_free(ggml_backend_t backend) {
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
