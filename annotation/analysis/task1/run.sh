#!/bin/bash
#SBATCH --gpus-per-node=2 --constraint=48GB

python agreement.py