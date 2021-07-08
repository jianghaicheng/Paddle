cc_library(ipu_backend SRCS ipu_backend.cc DEPS popart graph framework_proto enforce )
# cc_test(ipu_backend_test SRCS ipu_backend_test.cc DEPS ipu_backend)