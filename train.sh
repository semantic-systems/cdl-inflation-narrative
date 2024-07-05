#!/bin/bash
#SBATCH --gpus-per-node=2

huggingface-cli login --token $HFTOKENS
python sequence_classification.py one_hop_dag
python sequence_classification.py change_of_prices
python sequence_classification.py direction_of_change
