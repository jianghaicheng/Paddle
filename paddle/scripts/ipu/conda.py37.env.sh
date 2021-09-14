#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

IPU_PY_VER=3.7
export IPU_PY_VER=$IPU_PY_VER

PYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print('%s/%s' %
(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('INSTSONAME')))")

echo "PYTHON_LIBRARY: $PYTHON_LIBRARY"
export PYTHON_LIBRARY=$PYTHON_LIBRARY

PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")

echo "PYTHON_INCLUDE_DIR: $PYTHON_INCLUDE_DIR"
export PYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR

### build essentials

NOCMAKE_PATH=$PATH
PATH=$ROOT/paddle/scripts/ipu/installers/cmake-3.16.0-Linux-x86_64/bin:$PATH
export PATH
export NOCMAKE_PATH

### make sure use downloaded protoc used by c++ (v3.1.0)

# PATH=$ROOT/build/third_party/install/protobuf/bin:$PATH
# PATH=$ROOT/build/third_party/install/protobuf/include:$PATH
# export PATH
