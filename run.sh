#!/usr/bin/bash
docker run --runtime=nvidia -it -p 8888:8888 -v $PWD:/tmp -w /tmp tensorflow/tensorflow:latest-devel-gpu-py3
