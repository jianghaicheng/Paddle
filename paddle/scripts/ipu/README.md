# Paddle with IPU support

## Relevant tools

### gc-docker

`gc-docker` is the toolkit internally used to attach IPU devices for docker containers.

### gc-monitor

`gc-monitor` is used to watch IPU groups continously. We will use this tool to see whether a docker container is ready for development with IPU devices.

To query IPU you need to config your IPU host and parition id.

### poplar-sdk

Poplar-sdk is already shipped by GC team with our prebuilt docker image. You can access it in `/popsdk/` or `paddle/scripts/ipu/poplar-sdk`.

### Docker utils

To facilitate development wiht IPU, the codes is shipped with a dockerfile to build the environment.  

This dockerfile creates necessary environment to develop paddle with IPU support under root user \(will support non-root user by default soon\).

 Docker utils contains scripts to manage docker status for development

  ```bash
  # see how to use it
  docker_utils/into_docker.sh --help
  ```

### Other

We also add other relevant building scripts (will be integrated with dockerfile in the future) to build paddle and develop with IPU.

## Setup your IPU config  

- Set your IPU config
 
  ```bash
  mv envs.sh.template envs.sh
  ### IPU Config
  # Use `vipu list partition` to see partition ID
  export IPUOF_VIPU_API_PARTITION_ID="YOUR PARTITION ID"
  # Host name or IP address
  export IPUOF_VIPU_API_HOST="YOUR VIPU HOST"
  
  # Or equivalently, use:
  #   `mkdir -p /home/$USER/.ipuof.conf.d && \
  #       vipu get partition $IPUOF_VIPU_API_PARTITION_ID -H $IPUOF_VIPU_API_HOST --gcd 0 --ipu-configs >    /home/$USER/.ipuof.conf.d/your_pod.conf`
  #
  # to set up IPU servers
  #
  # export IPUOF_CONFIG_PATH="YOUR PATH TO '.ipuof.conf.d/your_podX_ipuof.conf'"
  ```
- Add SDK (inside docker container)
  
  ```bash
  source envs.sh
  ```

## Build Image

```bash
docker build -f Dockerfile.ipu.18.04 -t paddlepaddle/paddle:latest-ipu-dev .
```

or 

```bash
bash docker_utils/build_docker.sh
```

## Start a container with root user

If you want to create non-root user for successive tasks, you need first start container with root user:


```bash
bash docker_utils/into_docker.sh start --with-root

# add non-root user
cd $ROOT/paddle/scripts/ipu
bash docker_utils/add_user.sh

# install rdma to facilitate gc-monitor inside docker
bash installers/install_rdma.sh
```

We will simplify the process by modifying Dockerfile to support non-root user by default.

Later we need to install `rdma` to support `gc-monitor`:

```bash
bash installers/install_rdma.sh
```

## Start a non-root user

Once your create the non-root user, commit the change to new image, i.e. `$PADDLE_DOCKER_REPO:ipu-ubuntu18.04-gcc82-dev_v0.1`

Change relevant variable inside `into_docker.sh`:

- `IMG`:  
the image you want to run
- `CONTAINER_ID`:  
the container to start

```bash
bash docker_utils/into_docker.sh start
```

## Log into the started container

```bash
bash docker_utils/into_docker.sh into
```