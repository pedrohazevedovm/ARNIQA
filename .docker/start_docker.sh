#!/bin/bash
# Executa o container em background com nome fixo

docker run --gpus all -dit \
    --name pytorch_env_container \
    -v $(pwd):/workspace \
    --entrypoint /workspace/entrypoint.sh \
    -w /workspace \
    pytorch-env bash
