#!/usr/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -p exercise-gpu
#SBATCH --nodelist=csg-brook02

./run.py