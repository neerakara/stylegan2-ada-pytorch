#!/bin/bash
#
# Script to send job to SLURM clusters using sbatch.
# Usage: sbatch train.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SLURM Variables:
#SBATCH  --output=/data/scratch/nkarani/logs/%j.out
#SBATCH  --partition=titan,2080ti,gpu
#SBATCH  --exclude=malt,wasabi,fennel
#SBATCH  --gres=gpu:1
#SBATCH  --cpus-per-task=8
#SBATCH  --mem=12G
#SBATCH  --time=96:00:00
#SBATCH  --priority='TOP'

# ,curcum,sumac,fennel,urfa-biber,rosemary,juniper,cassia,marjoram
# activate virtual environment
source /data/vision/polina/users/nkarani/anaconda3/bin/activate env_density

## EXECUTION OF PYTHON CODE:
python /data/vision/polina/users/nkarani/projects/anomaly/ideas/stylegan/stylegan2-ada-pytorch/train.py \
--outdir='../training-runs/' \
--data='../datasets/fets256.zip' \
--gpus=1 \
--metrics='pr50k3_full' \
--aug='noaug'

echo "Hostname was: `hostname`"
echo "Reached end of job file."