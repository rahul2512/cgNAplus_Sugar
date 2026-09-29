#!/bin/bash -l
#SBATCH -J RYY
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00



python main.py RYY


