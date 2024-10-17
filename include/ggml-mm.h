#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_API ggml_backend_t ggml_backend_mm_init(void);

GGML_API bool ggml_backend_is_mm(ggml_backend_t backend);


#ifdef  __cplusplus
}
#endif
