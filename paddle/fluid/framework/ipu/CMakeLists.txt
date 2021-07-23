cc_library(ipu_utils SRCS ipu_utils.cc DEPS memory framework_proto popart)
cc_library(ipu_backend SRCS ipu_backend.cc DEPS popart graph framework_proto enforce ipu_utils)
