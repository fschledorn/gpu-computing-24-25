#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -p exercise-gpu
#SBATCH -o ex1_out.txt

bin/nullKernelAsync