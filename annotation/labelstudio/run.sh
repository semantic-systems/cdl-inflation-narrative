#!/bin/bash
#SBATCH --gpus-per-node=2 --constraint=48GB

python ./annotation/labelstudio/agreement.py