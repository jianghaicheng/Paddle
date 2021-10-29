set(POPART_CANONICALIZATION_SRC
    "popart_canonicalization/canonicalization_utils.cc"
    "popart_canonicalization/op_builder.cc"
    "popart_canonicalization/post_canonicalization.cc"
    "popart_canonicalization/activation_ops.cc"
    "popart_canonicalization/elementwise_ops.cc"
    "popart_canonicalization/logic_ops.cc"
    "popart_canonicalization/math_ops.cc"
    "popart_canonicalization/nn_ops.cc"
    "popart_canonicalization/tensor_ops.cc"
    "popart_canonicalization/other_ops.cc"
    "popart_canonicalization/reduce_ops.cc"
    "popart_canonicalization/search_ops.cc"
)

set(POPART_PATTERNS_SRC
    ""
)

set(POPART_OPERATORS_SRC
    ""
)

list(APPEND PADDLE_IPU_SRC ${POPART_CANONICALIZATION_SRC})
list(APPEND PADDLE_IPU_SRC ${POPART_PATTERNS_SRC})
list(APPEND PADDLE_IPU_SRC ${POPART_OPERATORS_SRC})
set(PADDLE_IPU_LIB "${CMAKE_CURRENT_BINARY_DIR}/libpaddle_ipu.so" CACHE STRING "")
set(PADDLE_IPU_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE STRING "")

set(IPU_BACKEND_SRC
    "device.cc"
    "ipu_strategy.cc"
    "ipu_optimizer.cc"
    "ipu_executor.cc"
    "ipu_compiler.cc"
    "ipu_backend.cc"
)

cc_library(ipu_utils SRCS ipu_utils.cc DEPS memory framework_proto popart)
cc_library(paddle_ipu SHARED SRCS ${PADDLE_IPU_SRC} DEPS popart ipu_utils)
cc_library(ipu_backend SRCS ${IPU_BACKEND_SRC} DEPS ipu_utils popart graph framework_proto enforce graph_helper timer)
