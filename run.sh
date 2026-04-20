#!/bin/bash
#SBATCH --gpus-per-node=2 --constraint=48GB

#python models/encoders_task_2.py
python models/llms_task_2_sft.py