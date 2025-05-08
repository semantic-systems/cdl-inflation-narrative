#!/bin/bash
#SBATCH --gpus-per-node=2 --constraint=48GB
#SBATCH --output=slurm-%j.out  # Speichert die Standardausgabe in einer Datei
#SBATCH --error=slurm-%j.err   # Speichert die Fehlerausgabe in einer Datei

python ./annotation/labelstudio/agreement.py