#include <slave.h>
#include <simd.h>
#include <stdint.h>
#include <stddef.h>
#include "tensors.h"
#define PT_MATH_RANGE_CHECKS
#include "pt_math.h"

#define remote_ldm_addr(coreid, ldm_var) (((unsigned long) &ldm_var | (unsigned long) (coreid) << 20) |(1ULL << 45))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    (void)(prefix##0);
#define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    (void)(prefix##1);
#define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    (void)(prefix##2);
#define GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    (void)(prefix##3);


#define ALIGNMENT_LDM 64
static __thread_local char __buffer[64 * 1024] __attribute__((aligned(ALIGNMENT_LDM)));
static int __buffer_i;

static inline void init(void) {
    __buffer_i = 0;
}

void* ldm_malloc_fast(size_t size) {
    void * res = (void *) &__buffer[__buffer_i];
    __buffer_i += size;
    __buffer_i += (ALIGNMENT_LDM - __buffer_i % ALIGNMENT_LDM) % ALIGNMENT_LDM;
    return res;
}

void mul_mat_fp32(void* args) {
    init();
    struct mul_mat_args* args_local = (struct mul_mat_args*) ldm_malloc_fast(sizeof(struct mul_mat_args));
    athread_dma_bcast_coll((void *) args_local, args, sizeof(struct mul_mat_args));

    struct athread_tensor* src0 = &args_local->src0;
    struct athread_tensor* src1 = &args_local->src1;
    struct athread_tensor* dst = &args_local->dst;

    GGML_TENSOR_LOCALS(int, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int, ne1, src1, ne)
    GGML_TENSOR_LOCALS(int, ne, dst, ne)

    GGML_TENSOR_LOCALS(size_t, nb0, src0, nb)
    GGML_TENSOR_LOCALS(size_t, nb1, src1, nb)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

    // broadcast factors
    const int r2 = ne12/ne02;
    const int r3 = ne13/ne03;
    float *y_row = (float *) ldm_malloc_fast(nb11);
    float *x_row = (float *) ldm_malloc_fast(nb01);
    floatv8 sum;
    float sum_value;
    athread_rply_t rply;

    for (int i13 = 0; i13 < ne13; i13++) {
        for (int i12 = 0; i12 < ne12; i12++) {
            const int i03 = i13 / r3;
            const int i02 = i12 / r2;

            char *x = ((char *) src0->data + i02 * nb02 + i03 * nb03);
            char *y = ((char *) src1->data + i12 * nb12 + i13 * nb13);
            char *d = ((char *) dst->data + i12 * nb2 + i13 * nb3);

            // Perform the matrix multiplication with initialization
            for (int i = 0; i < ne1; i++) {
                athread_dma_bcast_coll((void *) y_row, (void *) (y + nb11 * i), nb11);
                for (int j = _PEN; j < ne01; j +=64) {
                    sum = 0.0f;
                    athread_dma_get((void *) x_row, (void *) (x + nb01 * j), nb01);
                    for (int k = 0; k < ne10;k+=8) {
                        floatv8 v_x, v_y;
                        simd_load(v_x, &x_row[k]);
                        simd_load(v_y, &y_row[k]);
                        sum += v_x * v_y;
                    }
                    sum_value = simd_reduc_pluss(sum);
                    athread_dma_put((void *) (d + i * nb1 + j * nb0), (void*) &sum_value, sizeof(float));
                }
                athread_ssync_array();
            }
        }
    }
}

static __thread_local const int vshfw_c0_arr[16] = {0x29062080,0x16a4a1e6,0xee6b, 0,0,0,0,0,0,0,0,0,0,0,0,0};
static __thread_local const int vshfw_c1_arr[16] = {0xad2728c2,0x37ace3f6,0xfeef, 0,0,0,0,0,0,0,0,0,0,0,0,0};

void mul_mat_fp16(void* args) {
    init();
    struct mul_mat_args* args_local = (struct mul_mat_args*) ldm_malloc_fast(sizeof(struct mul_mat_args));
    athread_dma_bcast_coll((void *) args_local, args, sizeof(struct mul_mat_args));


    struct athread_tensor* src0 = &args_local->src0;
    struct athread_tensor* src1 = &args_local->src1;
    struct athread_tensor* dst = &args_local->dst;

    GGML_TENSOR_LOCALS(int, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int, ne1, src1, ne)
    GGML_TENSOR_LOCALS(int, ne, dst, ne)

    GGML_TENSOR_LOCALS(size_t, nb0, src0, nb)
    GGML_TENSOR_LOCALS(size_t, nb1, src1, nb)
    GGML_TENSOR_LOCALS(size_t, nb, dst, nb)


    const int STRIDE_NE01 = 16;
    // broadcast factors
    const int r2 = ne12/ne02;
    const int r3 = ne13/ne03;
    float *y_row = (float *) ldm_malloc_fast(nb11);
    float16 *x_row = (float16 *) ldm_malloc_fast(nb01);
    floatv8 sum;
    
    float sum_value[STRIDE_NE01];
    athread_rply_t rply;
    intv16 vshfw_c0,vshfw_c1;
    simd_load(vshfw_c0, (int *) vshfw_c0_arr);
    simd_load(vshfw_c1, (int *) vshfw_c1_arr);

    for (int i13 = 0; i13 < ne13; i13++) {
        for (int i12 = 0; i12 < ne12; i12++) {
            const int i03 = i13 / r3;
            const int i02 = i12 / r2;

            char *x = ((char *) src0->data + i02 * nb02 + i03 * nb03);
            char *y = ((char *) src1->data + i12 * nb12 + i13 * nb13);
            char *d = ((char *) dst->data + i12 * nb2 + i13 * nb3);

            // Perform the matrix multiplication with initialization
            for (int i = 0; i < ne1; i++) {
                athread_dma_bcast_coll((void *) y_row, (void *) (y + nb11 * i), nb11);
                for (int k = 0; k < ne10;k += 32) {
                    intv16 v0, v1;
                    simd_load(v0, (int *) &y_row[k +  0]);
                    simd_load(v1, (int *) &y_row[k + 16]);
                    intv16 v2, v3;
                    v2 = simd_vshfw(v0, v1, vshfw_c0);
                    v3 = simd_vshfw(v0, v1, vshfw_c1);
                    simd_store(v2, (int *) &y_row[k +  0]);
                    simd_store(v3, (int *) &y_row[k + 16]);
                }
                for (int j0 = _PEN * STRIDE_NE01; j0 < ne01; j0 += 64 * STRIDE_NE01) {
                    for (int j = j0, idx = 0; j < ne01 && idx < STRIDE_NE01; j ++, idx++) {
                        sum = 0.0f;
                        athread_dma_get((void *) x_row, (void *) (x + nb01 * j), nb01);
                        for (int k = 0; k < ne10; k += 32) {
                            float16v32 v_f16;
                            floatv8 v_f32;
                            simd_load(v_f16, &x_row[k]);

                            simd_load(v_f32, &y_row[k + 0]);
                            sum += v_f32 * simd_vfcvths(v_f16, 0);

                            simd_load(v_f32, &y_row[k + 8]);
                            sum += v_f32 * simd_vfcvths(v_f16, 1);

                            simd_load(v_f32, &y_row[k + 16]);
                            sum += v_f32 * simd_vfcvths(v_f16, 2);

                            simd_load(v_f32, &y_row[k + 24]);
                            sum += v_f32 * simd_vfcvths(v_f16, 3);
                        }
                        sum_value[idx] = simd_reduc_pluss(sum);
                    }
                    athread_dma_put((void *) (d + i * nb1 + j0 * nb0), sum_value, sizeof(float) * MIN(STRIDE_NE01, ne01 - j0));
                }
                athread_ssync_array();
            }
        }
    }
}

void cont_32(void* args) {
    init();
    struct unary_args* args_local = (struct unary_args*) ldm_malloc_fast(sizeof(struct unary_args));
    athread_dma_bcast_coll((void *) args_local, args, sizeof(struct unary_args));

    struct athread_tensor* src0 = &args_local->src0;
    struct athread_tensor* dst = &args_local->dst;
    char* dst_ptr = (char *) dst->data;
    char* src_ptr = (char *) src0->data;

    GGML_TENSOR_LOCALS(int, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t, nb0, src0, nb)

    int idx = 0;
    int i0, i1, i2, i3;

    if (nb00 == 4) {
        for (i3 = 0; i3 < ne03; i3++) {
            for (i2 = 0; i2 < ne02; i2++) {
                for (i1 = 0; i1 < ne01; i1++) {
                    if (idx == _PEN) {
#pragma GCC unroll 8
                        for (i0 = 0; i0 < ne00; i0++) {
                            *(uint32_t*) (dst_ptr + i0 * 4 + i1 * ne00 * 4 + i2 * ne00 * ne01 * 4 + i3 * ne00 * ne01 * ne02 * 4) = 
                            *(uint32_t*) (src_ptr + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03);
                        }
                    }
                    idx = ((idx + 1) & 0xff);
                }
            }
        }
    } else if (nb01 == 4) {
        for (i3 = 0; i3 < ne03; i3++) {
            for (i2 = 0; i2 < ne02; i2++) {
                if (idx == _PEN) {
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
                idx = ((idx + 1) & 0xff);
            }
        }
    } else {
        for (i3 = 0; i3 < ne03; i3++) {
            for (i2 = 0; i2 < ne02; i2++) {
                if (idx == _PEN) {
                    for (i1 = 0; i1 < ne01; i1++) {
                        for (i0 = 0; i0 < ne00; i0++) {
                            *(uint32_t*) (dst_ptr + i0 * 4 + i1 * ne00 * 4 + i2 * ne00 * ne01 * 4 + i3 * ne00 * ne01 * ne02 * 4) = 
                            *(uint32_t*) (src_ptr + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03);
                        }
                    }
                }
                idx = ((idx + 1) & 0xff);
            }
        }
    }
    flush_slave_cache();
}

void softmax_f32(void * args) {
    init();
    struct unary_args* args_local = (struct unary_args*) ldm_malloc_fast(sizeof(struct unary_args));
    athread_dma_bcast_coll((void *) args_local, args, sizeof(struct unary_args));

    struct athread_tensor* src0 = &args_local->src0;
    struct athread_tensor* dst = &args_local->dst;

    GGML_TENSOR_LOCALS(int, ne0, src0, ne)
    const int nc = ne00;
    const int nr = ne01 * ne02 * ne03;

    for (int i1 = _PEN; i1 < nr; i1+=64) {
        float * sp = (float *)((char *) src0->data + i1*src0->nb[1]);
        float * dp = (float *)((char *)  dst->data +  i1*dst->nb[1]);

        float max = -1e30;
        for (int j = 0;j < nc;j++) {
            max = MAX(max, sp[j]);
        }
        float sum = 0.0f;
        for (int j = 0;j < nc;j++) {
            dp[j] = (sp[j] - max) < -50.0f ? 0.0f : PT_expf(sp[j] - max);
            sum += dp[j];
        }
        sum = 1.0f/sum;
        for (int j = 0;j < nc;j++) {
            dp[j] = dp[j] * sum;
        }
    }
    flush_slave_cache();
}