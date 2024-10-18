#include <slave.h>
#include <simd.h>
#include <stdint.h>
#include <stddef.h>
#include "tensors.h"

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
static __thread_local char buffer[64 * 1024] __attribute__((aligned(ALIGNMENT_LDM)));
static int __buffer_i;

static inline void init(void) {
    __buffer_i = 0;
}

void* ldm_malloc_fast(size_t size) {
    void * res = (void *) &buffer[__buffer_i];
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
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;
    float *y_row = (float *) ldm_malloc_fast(nb11);
    float *x_row = (float *) ldm_malloc_fast(nb01);

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
                for (int j = _PEN; j < ne01; j+=64) {
                    floatv8 sum = 0.0f;
                    athread_dma_get((void *) x_row, (void *) (x + nb01 * j), nb01);
                    for (int k = 0; k < ne10;k+=8) {
                        floatv8 v_x, v_y;
                        simd_load(v_x, &x_row[k]);
                        simd_load(v_y, &y_row[k]);
                        sum += v_x * v_y;
                    }
                    float sum_value = simd_reduc_pluss(sum);
                    athread_dma_put((void *) (d + i * nb1 + j * nb0), (void*) &sum_value, sizeof(float));
                }
                athread_ssync_array();
            }
        }
    }
}

// static __thread_local const int vshfw_c0_arr[16] = {0xca130880,0x1b4a9038,0x0000bceb, 0,0,0,0,0,0,0,0,0,0,0,0,0};
// static __thread_local const int vshfw_c1_arr[16] = {0xda972988,0x9f6b987a,0x0000fefb, 0,0,0,0,0,0,0,0,0,0,0,0,0};

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


    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;
    float *y_row = (float *) ldm_malloc_fast(nb11);
    float16 *x_row = (float16 *) ldm_malloc_fast(nb01);
    // intv16 vshfw_c0,vshfw_c1;
    // simd_load(vshfw_c0, (int *) vshfw_c0_arr);
    // simd_load(vshfw_c1, (int *) vshfw_c1_arr);

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
                for (int k = 0; k < ne10;k+=32) {
                    // intv16 v0, v1;
                    // simd_load(v0, (int *) &y_row[k +  0]);
                    // simd_load(v1, (int *) &y_row[k + 16]);
                    // intv16 v2, v3;
                    // v2 = simd_vshfw(v0, v1, vshfw_c0);
                    // v3 = simd_vshfw(v0, v1, vshfw_c1);
                    // simd_store(v2, (int *) &y_row[k +  0]);
                    // simd_store(v3, (int *) &y_row[k + 16]);
                    float tmp[32];
                    for (int j = 0;j < 32;j++) {
                        tmp[j] = y_row[k + j];
                    }
                    for (int j = 0;j < 8;j++) {
                        y_row[k + j +  0] = tmp[4 * j + 0];
                        y_row[k + j +  8] = tmp[4 * j + 1];
                        y_row[k + j + 16] = tmp[4 * j + 2];
                        y_row[k + j + 24] = tmp[4 * j + 3];
                    }
                }
                for (int j = _PEN; j < ne01; j+=64) {
                    floatv8 sum = 0.0f;
                    float sum_value = 0.0f;
                    athread_dma_get((void *) x_row, (void *) (x + nb01 * j), nb01);
                    for (int k = 0; k < ne10;k+=32) {
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
                    sum_value = simd_reduc_pluss(sum);
                    athread_dma_put((void *) (d + i * nb1 + j * nb0), (void*) &sum_value, sizeof(float));
                }
                athread_ssync_array();
            }
        }
    }
}
