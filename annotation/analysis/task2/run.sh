#!/bin/bash
#SBATCH --gpus-per-node=2 --constraint=24GB

python analyze_task_2_annotation.py -p 11 12 13 14 -f