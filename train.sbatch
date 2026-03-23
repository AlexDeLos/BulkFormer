#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_ath_bulkformer
#SBATCH --output=/home/nfs/alexdelossanto/BulkFormer/logs_slurm/stdout-%x-%j.txt
#SBATCH --export=NONE

module purge
module use /opt/insy/modulefiles
module load miniconda/3.9
eval "$(conda shell.bash hook)"
conda activate bulkformer_env

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python3)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "=================================================="

cd /home/nfs/alexdelossanto/BulkFormer
python3 -u train_arabidopsis.py

echo "=================================================="
echo "Job finished"
echo "Date: $(date)"