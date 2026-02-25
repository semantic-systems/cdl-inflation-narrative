#!/bin/bash
#SBATCH --gpus-per-node=2 --constraint=48GB

#python encoders_task_2.py
python llms_task_2_sft.py