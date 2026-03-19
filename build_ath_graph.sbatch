#!/bin/bash

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=242:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --mail-type=END
#SBATCH --job-name=build_ath_graph
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
echo "=================================================="

# export PROJECT="/home/nfs/alexdelossanto/Dataset_fusion_Microarray"
# export BIG_STORAGE="/tudelft.net/staff-umbrella/GeneExpressionStorage"

# cd $PROJECT

python3 -u utils/build_graph_ara.py

echo "=================================================="
echo "Job finished"
echo "Date: $(date)"
