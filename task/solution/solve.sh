#!/bin/bash
set -e

# Copy the fixed solution files into /app and run the training
cp /workspaces/HandShake/task/solution/train_distributed.py /workspaces/HandShake/task/environment/app/train_distributed.py
mkdir -p /workspaces/HandShake/task/environment/app/utils
cp /workspaces/HandShake/task/solution/utils/checkpointing.py /workspaces/HandShake/task/environment/app/utils/checkpointing.py
touch /workspaces/HandShake/task/environment/app/utils/__init__.py

cd /workspaces/HandShake/task/environment/app
torchrun --nproc_per_node=2 train_distributed.py
