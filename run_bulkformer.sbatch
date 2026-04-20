#!/bin/bash
#SBATCH --job-name=bulkformer_viz
#SBATCH --account=testusers
#SBATCH --partition=gpu          # Use the GPU partition
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --time=04:00:00          # 4 hours should be plenty with a GPU
#SBATCH --mem=32G                # High RAM for pandas, graph loading, and UMAP
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/alexdelossanto/linuxhome/Dataset_fusion_Microarray/logs_slurm/bulkformer-%x-%j.txt

# ══════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════
BIG_STORAGE="/tudelft.net/staff-umbrella/GeneExpressionStorage"
SIF="$BIG_STORAGE/bulkformer.sif"
REPO_DIR="/home/alexdelossanto/linuxhome/Dataset_fusion_Microarray"

echo "=================================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $(hostname)"
echo "Date:            $(date)"
echo "SIF:             ${SIF}"
echo "=================================================="

# ══════════════════════════════════════════════════
#  CREATE HYBRID PASSWD/GROUP FILES FOR CONTAINER
# ══════════════════════════════════════════════════
MY_PASSWD="$BIG_STORAGE/tmp_passwd_$SLURM_JOB_ID"
MY_GROUP="$BIG_STORAGE/tmp_group_$SLURM_JOB_ID"

cat /etc/passwd > "$MY_PASSWD"
getent passwd $(whoami) >> "$MY_PASSWD"

cat /etc/group > "$MY_GROUP"
for g in $(id -G -n); do
    getent group "$g" >> "$MY_GROUP" || true
done
sort -u "$MY_GROUP" -o "$MY_GROUP"

# ══════════════════════════════════════════════════
#  RUN THE PYTHON SCRIPT INSIDE THE CONTAINER
# ══════════════════════════════════════════════════
# Note the addition of the --nv flag to pass the Nvidia GPU into the container
apptainer exec \
    --nv \
    --bind /lib64/libsubid.so.3:/usr/lib64/libsubid.so.3 \
    --bind /lib64/libcrypt.so.2:/usr/lib64/libcrypt.so.2 \
    --bind "$MY_PASSWD":/etc/passwd \
    --bind "$MY_GROUP":/etc/group \
    --bind /tudelft.net/staff-umbrella:/tudelft.net/staff-umbrella \
    --bind "$REPO_DIR":/repo \
    --pwd /repo \
    "$SIF" \
    python3 -u /repo/run_bulkformer_viz.py

# ══════════════════════════════════════════════════
#  CLEANUP
# ══════════════════════════════════════════════════
echo "Cleaning up all temporary directories..."
rm -f "$MY_PASSWD" "$MY_GROUP"

echo "=================================================="
echo "Job Completed"
echo "=================================================="