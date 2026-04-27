#!/bin/bash
#SBATCH --gpus-per-node=1 --constraint=48GB

#python encoders_task_2.py 
CUDA_VISIBLE_DEVICES=1 python inference.py