# Configure CI for workflow

```bash
# build docker image
docker build -t paddlepaddle/paddle:latest-dev-ipu \
-f tools/dockerfile/Dockerfile.ipu .

docker volume create paddle_ccache
docker volume create --driver local \
  --opt type=none \
  --opt device={DIR_TO_IPU_OF} \
  --opt o=bind \
  paddle_ipuof
# where has {DIR_TO_IPU_OF}/ipu.conf

docker volume create --driver local \
  --opt type=none \
  --opt device={DIR_TO_PADDLE_WHEELS} \
  --opt o=bind \
  paddle_wheels
# {DIR_TO_PADDLE_WHEELS} is where the python package `paddle_*.whl` saved

# check configure
docker run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK \
--device=/dev/infiniband/ --ipc=host \
-v paddle_ccache:/paddle_ccache \
-e CCACHE_DIR=/paddle_ccache \
-e CCACHE_MAXSIZE=30G \
-v paddle_ipuof:/ipuof \
-v paddle_wheels:/paddle_wheels \
-e IPUOF_CONFIG_PATH=/ipuof/ipu.conf \
--rm \
paddlepaddle/paddle:latest-dev-ipu \
bash -c "pwd & gc-monitor && ccache -s"

# set up github action-runner
# run action-runner in tmux session
```
