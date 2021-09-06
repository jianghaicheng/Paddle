cd /paddle/build
source /popsdk/poplar-ubuntu_18_04-2.1.0+145366-ce995e299d/enable.sh
source /popsdk/popart-ubuntu_18_04-2.1.0+145366-ce995e299d/enable.sh
cmake .. -DPY_VERSION=3.7 \
          -DWITH_GPU=OFF \
          -DWITH_TESTING=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          -DWITH_NCCL=OFF \
          -DWITH_RCCL=OFF \
          -DWITH_IPU=ON \
          -DPOPLAR_SDK_DIR=/popsdk/
make -j
pip install -U ./python/dist/paddlepaddle*
