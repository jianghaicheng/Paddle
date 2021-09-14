# 在 IPU 上运行 Paddle

## 相关工具

### gc-docker

`gc-docker` 是内部加载 ipu 设备到 docker容器的工具。

### gc-monitor

`gc-monitor` 用于持续监控 IPU 运行。我们将用这个工具检查容器是否已经准备好IPU上的开发环境。

### poplar-sdk

`Poplar-sdk` 已经包含在 GraphCore 团队预先构建的 docker 镜像内。可以在 `/popsdk/` 或者 `paddle/scripts/ipu/poplar-sdk` 目录下访问相关文件。

### Docker utils  

为了方便在IPU上开发，代码已经包含了构建所需环境的 Dockerfile 文件。

用 Dockerfile 文件可以创建在IPU上开发的必须环境，默认是root用户（稍后我们会支持默认非root用户）。

`Docker utils` 包含了管理 docker 状态的脚本用于开发工作：

  ```bash
  # see how to use it
  docker_utils/into_docker.sh --help
  ```

### 其他

我们同事添加其他必须的构建脚本用于在IPU上开发（稍后会集成在dockerfile）

## 配置 IPU   

- 配置IPU
 
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
- 添加 SDK (在容器内)
  
  ```bash
  source envs.sh
  ```

## 构建镜像

```bash
docker build -f Dockerfile.ipu.18.04 -t paddlepaddle/paddle:latest-ipu-dev .
```

或者

```bash
bash docker_utils/build_docker.sh
```

## Root 用户下启动容器

如果需要创建非 root 用户，需要首先 root 用户下启动容器：


```bash
bash docker_utils/into_docker.sh start --with-root

# add non-root user
cd $ROOT/paddle/scripts/ipu
bash docker_utils/add_user.sh

# install rdma to facilitate gc-monitor inside docker
bash installers/install_rdma.sh
```

稍后我们会在 Dockerfile 简化对非root用户的支持

接着我们需要安装 `rdma` 用于支持 `gc-monitor`

```bash
bash installers/install_rdma.sh
```

## 非 Root 用户下启动容器

创建非root用户后，更新提交到镜像，比如 `$PADDLE_DOCKER_REPO:ipu-ubuntu18.04-gcc82-dev_v0.1`

更改 `into_docker.sh` 相关变量：

- `IMG`:  
镜像名称
- `CONTAINER_ID`:  
容器名称

```bash
bash docker_utils/into_docker.sh start
```

## 进入启动后的容器

```bash
bash docker_utils/into_docker.sh into
```