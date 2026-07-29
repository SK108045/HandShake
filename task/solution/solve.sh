#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$DIR")")"

# Copy the fixed solution files into /app and run the training
cp "$DIR/train_distributed.py" "$ROOT_DIR/task/environment/app/train_distributed.py"
mkdir -p "$ROOT_DIR/task/environment/app/utils"
cp "$DIR/utils/checkpointing.py" "$ROOT_DIR/task/environment/app/utils/checkpointing.py"
touch "$ROOT_DIR/task/environment/app/utils/__init__.py"

cd "$ROOT_DIR/task/environment/app"
torchrun --nproc_per_node=2 train_distributed.py
