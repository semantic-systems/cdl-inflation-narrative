#!/bin/bash
#SBATCH --gpus-per-node=2 --constraint=48GB

python models/encoders_task_2.py