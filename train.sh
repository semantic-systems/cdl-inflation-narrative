#!/bin/bash
#SBATCH --gpus-per-node=2

huggingface-cli login --token $HFTOKENS
python sequence_classification.py core_event
