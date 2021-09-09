#!/usr/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"

source $ROOT/paddle/scripts/ipu/utils.sh

BUILD=${ROOT}/build
# ${ROOT}/cmake-build-debug
mkdir -p $BUILD

cd $BUILD
# used in find poplar, popart directories
set -x
if [ $POPLAR_SDK_DIR == "" || ! -z "$POPLAR_SDK_DIR" ];then
  if [ -d "$ROOT/paddle/scripts/ipu/poplar-sdk/poplar_sdk-ubuntu_18_04-2.1.0+617-6bb5f5b742" ];then
    POPLAR_SDK_DIR=$ROOT/paddle/scripts/ipu/poplar-sdk/poplar_sdk-ubuntu_18_04-2.1.0+617-6bb5f5b742/
  elif [ -d /popsdk/ ];then
    POPLAR_SDK_DIR=/popsdk/
    # enable Poplar SDK for jekin machine
    source /popsdk/poplar-ubuntu_18_04-2.1.0+145366-ce995e299d/enable.sh
    source /popsdk/popart-ubuntu_18_04-2.1.0+145366-ce995e299d/enable.sh
  else
    err "Could not find valid popsdk directoriy!"
    exit 1
  fi
fi
set +x
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
