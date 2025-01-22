#!/bin/env bash
#SBATCH --partition=scavenge-lg
#SBATCH --account=scavenge-lg
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --output=ads.out
#SBATCH --mail-user=gmli@isi.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,ALL
#SBATCH --open-mode=truncate

python3 test2.py