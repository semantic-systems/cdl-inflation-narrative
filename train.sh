#!/bin/bash
#SBATCH --gpus-per-node=2

conda activate cdl
python sequence_classification.py
conda deactivate
