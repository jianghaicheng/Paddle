set(POPART_CANONICALIZATION_HANDLERS_SRC
    "popart_canonicalization/other_ops.cpp"
)
# TODO(alleng) build static library
cc_library(popart_canonicalization_utils SHARED SRCS popart_canonicalization_utils.cc 
  ${POPART_CANONICALIZATION_HANDLERS_SRC} DEPS framework_proto enforce)

cc_library(ipu_utils SRCS ipu_utils.cc DEPS memory framework_proto popart)
cc_library(ipu_backend SRCS ipu_backend.cc DEPS popart graph framework_proto enforce ipu_utils)
