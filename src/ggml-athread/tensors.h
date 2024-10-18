#define GGML_MAX_DIMS 4

struct athread_tensor {
    size_t nb[GGML_MAX_DIMS]; // stride in bytes:
    int ne[GGML_MAX_DIMS];    // number of elements
    void* data;
};

struct mul_mat_args {
    struct athread_tensor dst;
    struct athread_tensor src0;
    struct athread_tensor src1;
};
