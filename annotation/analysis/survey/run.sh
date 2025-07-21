#!/bin/bash
#SBATCH --gpus-per-node=1 --constraint=24GB

python task_2_agreement.py -f