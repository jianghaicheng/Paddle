#!/usr/bin/env bash

echo "BASH_SOURCE[0]: ${BASH_SOURCE[0]}"
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

pushd $ROOT/poplar-sdk

# @todo TODO(yiakw) : change version
poplar_tar="poplar_sdk-ubuntu_18_04-2.1.0+617-6bb5f5b742"
export POPLAR_SDK_DIR=$poplar_tar

source poplar-sdk.env.sh

source $poplar_tar/popart-ubuntu_18_04-2.1.0+145366-ce995e299d/enable.sh
source $poplar_tar/poplar-ubuntu_18_04-2.1.0+145366-ce995e299d/enable.sh
export TF_POPLAR_BASE=$poplar_tar/popart-ubuntu_18_04-2.1.0+145366-ce995e299d

popd

