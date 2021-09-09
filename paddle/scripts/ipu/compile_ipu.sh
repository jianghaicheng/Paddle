#!/usr/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"

source utils.sh

BUILD=${ROOT}/build
# ${ROOT}/cmake-build-debug
mkdir -p $BUILD

cd $BUILD
# used in find poplar, popart directories
if [ ! -z "${POPLAR_SDK_DIR}" ];then
  if [ -d "$ROOT/paddle/scripts/ipu/poplar-sdk/poplar_sdk-ubuntu_18_04-2.1.0+617-6bb5f5b742" ];then
    POPLAR_SDK_DIR=$ROOT/paddle/scripts/ipu/poplar-sdk/poplar_sdk-ubuntu_18_04-2.1.0+617-6bb5f5b742/
  elif [ -d /popskd/ ];then
    POPLAR_SKD_DIR=/popsdk/
  else
    err "Could not find valid popsdk directoriy!"
    exit 1
  fi
fi

cmake .. -DPY_VERSION=3.7 \
          -DWITH_GPU=OFF \
          -DWITH_TESTING=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_NCCL=OFF \
          -DWITH_RCCL=OFF \
          -DWITH_IPU=ON \
          -DPOPLAR_SDK_DIR="$POPLAR_SDK_DIR"
make -j
pip install -U $BUILD/python/dist/paddlepaddle*
