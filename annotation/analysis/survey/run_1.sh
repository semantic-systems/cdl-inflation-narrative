#!/bin/bash
#SBATCH --gpus-per-node=1

python task_1_agreement.py #-p 20 21 -f --redownload