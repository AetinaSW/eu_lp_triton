#!/bin/bash

sudo apt-get install -y python3-pip
pip3 install launchpadlib==1.10.13
pip3 install setuptools==59.5.0
pip3 install grpcio
pip3 install grpcio-tools
pip3 install tritonclient[all]

