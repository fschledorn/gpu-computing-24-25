#!/usr/bin/env bash

make && srun -p exercise-gpu --gres=gpu:1 -o ex2_out.txt bin/breakeven
