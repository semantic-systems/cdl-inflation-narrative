#!/bin/bash
#SBATCH --gpus-per-node=1 --constraint=24GB

python analyze_elfen.py
