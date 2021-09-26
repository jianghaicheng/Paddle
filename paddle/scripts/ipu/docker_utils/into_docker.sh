#/usr/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.." && pwd )"

set -e

# include libraries
source $ROOT/paddle/scripts/ipu/utils.sh

# predefined reposiotry and images
# @todo TODO(yiakwy) : add prebuild docker image repo
DOCKER_REPO=graph-core
PADDLE_DOCKER_REPO=paddlepaddle/paddle
# simliar to "nvidia-docker", here we use gc-docker (see poplar-sdk)
DOCKER_CMD=gc-docker

if ! [ -x "$(command -v ${DOCKER_CMD})" ]; then
  warn "DOCKER_CMD=$DOCKER_CMD not found! Set to docker"
  DOCKER_CMD="docker"
fi

IMG=$PADDLE_DOCKER_REPO:ipu-ubuntu18.04-gcc82-dev_v0.1

# container id
CONTAINER_ID=paddle_ipu_dev_v0.1

# used to update from image registry
VERSION=

# only echo command used for debug
DRY_RUN=false

# docker arguments
DOCKER_ARGS=""

# created containers
containers=

# docker user
docker_user=

# external disks
external_disks=

get_containers() {
  containers=$(docker container ls -a --format "{{.Names}}")
  echo "created containers : "
  for c in ${containers[@]}; do
   echo "CONTAINER: $c"
  done
}

# light wrapper for docker to create an image
_run_container_with_root() {
  local container_id=$1
  local img=$2
  local docker_args=
 
  USER_ID=$(id -u) # 0 for root user
  GRP=$(id -g -n)
  GRP_ID=$(id -g) # 0 for root user
  DOCKER_HOME="/home/$USER"

  docker_args=$(echo --privileged=true -it \
    --name $container_id \
    -e CONTAINER_NAME=$container_id \
    -e DOCKER_USER=$USER \
    -e USER=$USER \
    -e DOCKER_USER_ID=$USER_ID \
    -e DOCKER_GRP=$GRP \
    -e DOCKER_GRP_ID=$GRP_ID \
    -e LC_ALL='' \
    -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK \
    ${DOCKER_ARGS[@]} \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/resolv.conf:/etc/resolv.conf:ro \
    -v /etc/hosts:/etc/hosts:ro \
    -v /run/user/$USER_ID/keyring/ssh:/ssh-agent:rw \
    -v /dev/snd:/dev/snd:rw \
    --group-add sudo \
    --workdir "/home" \
    $img /bin/bash)

  docker_args=($docker_args)

  if [ $DOCKER_CMD == "gc-docker" ]; then
    set -x
    $DOCKER_CMD -- "${docker_args[@]}"
    set +x
  else
    set -x
    $DOCKER_CMD run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK --device=/dev/infiniband/ "${docker_args[@]}"
    set +x
  fi 
}

_run_container() {
  local container_id=$1
  local img=$2
  local is_container_created=
  local docker_args=

  USER_ID=$(id -u) # 0 for root user
  GRP=$(id -g -n) 
  GRP_ID=$(id -g) # 0 for root user
  DOCKER_HOME="/home/$USER"

  # @todo TODO(yiakwy) : will be support soon
  # RUNTIME="ipu"

  # ADD HOST NAME
  HOSTNAME="$HOSTNAME-in_docker_dev"

  # SHARE Memory Mode: just for development purpose, not safe for hosting public services
  IPC_NAMESPACE=host

  # check if the container has already been booted
  get_containers

  is_container_created=false  
  for c in ${containers[@]}; do
    if [[ "$c" == "$container_id" ]]; then
      is_container_created=true
      break
    fi
  done

  info "is_container_created: $is_container_created"
  if [ "$is_container_created" == "false" ]; then
    echo "creating container $container_id with image $img"
    echo "use gc-monitor see IPU devices"
    # @todo TODO(yiakwy) : enable to mount user space with fully permissions
    docker_args=$(echo -it \
      -d \
      --privileged \
      --name="$container_id" \
      -e CONTAINER_NAME=$container_id \
      -e DOCKER_USER=$USER \
      -e USER=$USER \
      -e DOCKER_USER_ID=$USER_ID \
      -e DOCKER_GRP=$GRP \
      -e DOCKER_GRP_ID=$GRP_ID \
      -e PADDLE_INSTALL='/paddle' \
      -e LC_ALL='' \
      -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK \
      -e HOME=/home/$USER \
      -u $USER_ID:$GRP_ID \
      ${DOCKER_ARGS[@]} \
      -v /media:/media \
      -v /etc/localtime:/etc/localtime:ro \
      -v /etc/timezone:/etc/timezone:ro \
      -v /etc/resolv.conf:/etc/resolv.conf:ro \
      -v /etc/hosts:/etc/hosts:ro \
      -v /etc/passwd:/etc/passwd:ro \
      -v /etc/group:/etc/group:ro \
      -v /etc/sudoers.d:/etc/sudoers.d:ro \
      -v /etc/sudoers:/etc/sudoers:ro \
      -v /run/user/$USER_ID/keyring/ssh:/ssh-agent:rw \
      -v /home/$USER:/home/$USER:rw \
      ${COMMENT# (yiakwy: not work): -v $HOME/.cache:${DOCKER_HOME}/.cache:rw} \
      -v /dev/snd:/dev/snd:rw \
      --ipc=$IPC_NAMESPACE \
      --add-host $HOSTNAME:127.0.0.1 \
      --add-host `hostname`:127.0.0.1 \
      --hostname $HOSTNAME \
      --group-add sudo \
      --workdir "$HOME" \
      $img /bin/bash) 

    docker_args=($docker_args)

    if [ $DOCKER_CMD == "gc-docker" ]; then
      set -x
      $DOCKER_CMD -- "${docker_args[@]}"
      set +x
    else
      set -x
      $DOCKER_CMD run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK --device=/dev/infiniband/ "${docker_args[@]}"
      set +x  
    fi 
  fi
 
  echo "starting container $container_id"
  
  if [ "$DRY_RUN" == "true" ]; then
    echo "docker container start -a -i $container_id"
  else 
    docker container start -a -i $container_id
  fi
}

_into_container() {
  local container_id=$1
  args=(
  )
  # xhost +local:root 1>/dev/null 2>&1
  if [ "$DRY_RUN" == "true" ]; then
    echo "docker exec -it -u $USER $container_id /bin/bash"
  else 
    docker exec -it -u $USER $container_id /bin/bash
  fi
  # xhost -local:root 1>/dev/null 2>&1
}

run_container() {
  if [ "$docker_user" == "root" ]; then
    _run_container_with_root  $CONTAINER_ID $IMG
  else
    _run_container $CONTAINER_ID $IMG
  fi
}

into_container() {
  info "into $CONTAINER_ID"
  _into_container $CONTAINER_ID
}

usage() {
  echo "Usage: $0 [OPTIONS] CMD"
  echo
  echo "docker utilties for Graph Core IPU users"
  echo
  echo "Options:"
  echo "    --with-root                start docker with root user without binding user space /home/$USER"
  echo "    --dry-run                  show commands to execute without executing"
  echo "    -- docker_args             pass addition docker_args to docker command"
  echo
  echo "Supported command:"
  echo
  echo "start                          run or start a container from $IMG build or pull from build_docker.sh"
  echo "into                           go into docker with user $USER"
}

main() {
  local args=()
  local sub_cmd=
  local docker_user=
  if [ $# -lt 1 ];then
     err "At least one command should be given"
     usage
     exit 1
  fi

  if ! [ "$EUID" -ne 0 ]; then
     err "$0: should not be root" 1>&2
     exit 1
  fi

  # parse options
  while [[ "$#" -gt 0 ]]; do
    case $1 in 
      --with-root) docker_user=root; shift ;;
      -h|--help) 
        usage  
        exit 1 
        ;;
      --dry-run) 
        DRY_RUN=true
        DOCKER_CMD="echo $DOCKER_CMD"
        shift 
        ;;
      --) 
        DOCKER_ARGS="$DOCKER_ARGS ${@:2}"
        break 2
        ;;
      *)
        args+=("$1") 
        shift
        ;;
    esac
  done

  if [ ${#args[@]} -ne 1 ];then
     err "Please input a command!"
     usage
     exit 1
  fi

  # parse CMD
  sub_cmd=${args[0]}
  case $sub_cmd in 
    start)
      run_container "${@:2}"
      ;;
    into)
      into_container "${@:2}"
      ;;
    *)
      usage
      err "Not supported command <$sub_cmd>!; see help. Pull requests are welcome!"
      ;;
  esac

}

main "$@"
