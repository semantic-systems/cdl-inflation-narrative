#!/bin/bash
#SBATCH --gpus-per-node=2 --constraint=48GB

huggingface-cli login --token $HFTOKENS
python sequence_classification.py inflation
