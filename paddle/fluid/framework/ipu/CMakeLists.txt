set(POPART_CANONICALIZATION_SRC
    "popart_canonicalization/canonicalization_utils.cc"
    "popart_canonicalization/activation_ops.cc"
    "popart_canonicalization/logic_ops.cc"
    "popart_canonicalization/math_ops.cc"
    "popart_canonicalization/nn_ops.cc"
    "popart_canonicalization/tensor_ops.cc"
    "popart_canonicalization/other_ops.cc"
)
cc_library(popart_canonicalization_utils SRCS ${POPART_CANONICALIZATION_SRC} DEPS framework_proto enforce)

cc_library(ipu_device SRCS device.cc DEPS enforce popart)
cc_library(ipu_utils SRCS ipu_utils.cc DEPS memory framework_proto popart)
cc_library(ipu_build_strategy SRCS ipu_build_strategy.cc DEPS popart graph framework_proto enforce)
cc_library(ipu_backend SRCS ipu_backend.cc DEPS popart graph framework_proto enforce ipu_utils ipu_build_strategy ipu_device graph_helper)
