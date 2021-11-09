# Configure CI for workflow

```bash
# setup ENV
# source POPLAR_SDK/enable.sh
# source POPART_SDK/enable.sh
# if no image graphcore/poplar:2.3.0
# download from https://downloads.graphcore.ai/
# build docker image
docker build -t paddle_ipu_ci:2.3.0 -f paddle/scripts/ipu/github_action/Dockerfile.poplar.2.3.0 .

docker volume create paddle_ccahe
docker volume create --driver local \
  --opt type=none \
  --opt device={DIR_TO_IPU_OF} \
  --opt o=bind \
  paddle_ipuof
# where has {DIR_TO_IPU_OF}/ipu.conf

# check configure
gc-docker -- \
-v paddle_ccahe:/paddle_ccahe \
-e CCACHE_DIR=/paddle_ccahe \
-e CCACHE_MAXSIZE=20G \
-v paddle_ipuof:/ipuof \
-e IPUOF_CONFIG_PATH=/ipuof/ipu.conf \
--rm \
paddle_ipu_ci:2.3.0 \
bash -c "pwd & gc-monitor && ccache -s"
# install github action-runner
# https://github.com/graphcore/Paddle_internal/settings/actions/runners/new
# run action-runner in tmux session
```
