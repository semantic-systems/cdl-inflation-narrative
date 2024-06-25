#!/bin/bash
#SBATCH --gpus-per-node=2

conda activate cdl
python train.py
conda deactivate
