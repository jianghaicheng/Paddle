#!/usr/bin/env bash
set -e

# computing absolute root
echo "$BASH_SOURCE"
ROOT="$( cd "$( dirname "$BASH_SOURCE[0]" )/../../../../" && pwd )"
echo "ROOT: $ROOT"

if ! [ -z "${DOCKER_REPO}" ];then
  DOCKER_REPO=gc-core
fi
PADDLE_DOCKER_REPO=paddlepaddle/paddle

TIME=""
ARCH=
TAG=
USE_PROXY="FALSE"
DOCKER_FILE=
# similar to "nvidia-docker", later we will use "gc-docker" (see poplar-sdk)
DOCKER_CMD=docker

Ubuntu_() {
  TIME=$(date +%Y%m%d_%H%M)
  ARCH=$(uname -m)
  TAG="dev-${ARCH}-${TIME}_$1"
}

build_img() {
  tag_suffix=$2
  Ubuntu_ $tag_suffix

  # set network proxy
  # if you are experiencing a slow network, consider arg UBUNTU_MIRROR and set it to the nearest endpoint.
  # HTTP_PROXY="http://127.0.0.1:8888"
  # HTTPS_PROXY="https://127.0.0.1:8888"
  args=(
  "--build-arg http_proxy=$HTTP_PROXY"
  "--build-arg https_proxy=$HTTPS_PROXY"
  )

  DOCKER_FILE=$1

  # to avoid bad influences from intermediate layers
  # remove dangling images : docker rmi -f $(docker images -f "dangling=true" -q)
  if [ USE_PROXY = "YES" ]; then
    echo "Using proxy HTTP_PROXY=$HTTP_PROXY HTTPS_PROXY=$HTTPS_PROXY"
    $DOCKER_CMD build -t "$PADDLE_DOCKER_REPO:$TAG" \
      -f "$ROOT/paddle/scripts/ipu/$DOCKER_FILE" \
      "$ROOT" $args
  else
    $DOCKER_CMD build -t "$PADDLE_DOCKER_REPO:$TAG" \
      -f "$ROOT/paddle/scripts/ipu/$DOCKER_FILE" \
      "$ROOT"
  fi

}

# @todo TODO(yiakwy)
update_img() {
  echo "Not Implemented Yet!"
}

# @todo TODO(yiakwy)
fetch_img() {
  echo "Not Implemented Yet!"
}

upload_img() {
  echo "Not Implemented Yet!"
}

main() {
# do not modify the lines below  unless you understand what you are doing ...

build_img "Dockerfile.ipu.18.04" "ipu-ubuntu18.04-gcc82"
}

main
