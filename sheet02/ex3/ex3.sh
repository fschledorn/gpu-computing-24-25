#!/usr/bin/env bash

make && srun -p exercise-gpu --gres=gpu:1 -o ex3_out.csv bin/timeMemcpy
python plot.py