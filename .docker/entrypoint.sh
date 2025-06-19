#!/usr/bin/env bash

source /opt/conda/etc/profile.d/conda.sh
conda activate myenv

mkdir arniqa
cp -r phavm/arniqa/ arniqa/

if [ $# -gt 0 ];then
    # If we passed a command, run it
    exec "$@"
else
    /bin/bash
fi
