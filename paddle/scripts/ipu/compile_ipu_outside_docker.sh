#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

BUILD=${ROOT}/build 
# ${ROOT}/cmake-build-debug
mkdir -p $BUILD

cd $BUILD

# Support build Paddle IPU branch outside of docker container
#
#   This will use special toollkit gcc-8.2 and cmake-3.16 to build paddle 2.1 with IPU support
#
# Addtional flags to add
#  -DON_INERENCE=ON
#  -DProtobuf_INCLUDE_DIR=${BUILD}/third_party/install/protobuf/include
PATH=/usr/local/gcc-8.2/bin:$PATH cmake .. -DPYTHON_EXECUTABLE:FILEPATH=$(which python) \
          -DPYTHON_INCLUDE_DIR:PATH=$PYTHON_INCLUDE_DIR \
          -DPYTHON_LIBRARY:FILEPATH=$PYTHON_LIBRARY \
          -DWITH_GPU=OFF \
          -DWITH_TESTING=OFF \
          -DWITH_NCCL=OFF \
          -DWITH_RCCL=OFF \
         `-DProtobuf_INCLUDE_DIR=${BUILD}/third_party/install/protobuf/include` \
          -DCMAKE_BUILD_TYPE=Release

make -j
pip install -U $BUILD/python/dist/*.whl
