#!/bin/bash

#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8

singularity run --nv -B /data:/data -B /scratch:/scratch /data/cs3450/pytorch20.11.3.sif python run_cifar.py