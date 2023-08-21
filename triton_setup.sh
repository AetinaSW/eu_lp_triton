#!/bin/bash
################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

set -e

TRITON_DOWNLOADS=./tritonserver
TRITON_PKG_PATH=${TRITON_PKG_PATH:=https://github.com/triton-inference-server/server/releases/download/v2.30.0/tritonserver2.30.0-jetpack5.1.tgz}

echo "Installing Triton prerequisites ..."
if [ $EUID -ne 0 ]; then
    echo "Must be run as root or sudo"
    exit 1
fi

apt-get update && \
    apt-get install -y --no-install-recommends libb64-dev libre2-dev libopenblas-dev

echo "Creating ${TRITON_DOWNLOADS} directory ..."
mkdir -p $TRITON_DOWNLOADS

echo "Downloading ${TRITON_PKG_PATH} to ${TRITON_DOWNLOADS} ... "
wget -O $TRITON_DOWNLOADS/jetpack.tgz $TRITON_PKG_PATH

echo "Extracting the package ....."
pushd $TRITON_DOWNLOADS
tar -xvf jetpack.tgz
rm jetpack.tgz

echo "Convert the model ......"
chmod 777 -R tao-converter/tao-converter
chmod 777 -R ./*.sh
./model_convert.sh


popd


