#!/bin/bash
#!/bin/bash
#SBATCH --account=testusers
#SBATCH --partition=all
#SBATCH --time=4:00:00
#SBATCH --mem=320G
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/alexdelossanto/linuxhome/Dataset_fusion_Microarray/logs_slurm/stdout-%x-%A_%a.txt
# ── User configuration ─────────────────────────────────────────────────────────
# Set this to the actual location of your .sif file on the shared storage.
# It MUST be on /tudelft.net/... — paths under /home/ are not visible on compute nodes.
REPO_DIR="/home/alexdelossanto/linuxhome/Dataset_fusion_Microarray"
BIG_STORAGE="/tudelft.net/staff-umbrella/GeneExpressionStorage"
SIF="$BIG_STORAGE/fused.sif"
# ──────────────────────────────────────────────────────────────────────────────

# Make sure the log directory exists before the job starts writing to it
mkdir -p "$(dirname "$SLURM_STDOUT")" 2>/dev/null || true

echo "========================================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo "SIF         : ${SIF}"
echo "========================================"

# Verify the .sif file is reachable from this compute node before attempting exec.
# If this fails the path is wrong — check with: find /tudelft.net -name "*.sif" 2>/dev/null
if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container image not found at ${SIF}"
    echo "Available .sif files under ${STORAGE}:"
    find "${STORAGE}" -name "*.sif" 2>/dev/null
    exit 1
fi

apptainer exec --nv \
    --bind "$BIG_STORAGE":"$BIG_STORAGE" \
    --bind "$REPO_DIR":/repo \
    --pwd /repo \
    "$SIF" \
    python train_arabidopsis.py

EXIT_CODE=$?
echo "========================================"
echo "Finished    : $(date)"
echo "Exit code   : ${EXIT_CODE}"
echo "========================================"
exit ${EXIT_CODE}
