#!/usr/bin/env bash

sudo apt-get update

# RDMA is used to help gc-monitor to find IPUs inside docker
sudo apt-get install -y \
	rdma-core \
	librdmacm1 \
	ibverbs-utils

