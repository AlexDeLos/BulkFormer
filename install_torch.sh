#!/bin/bash
# install_torch.sh
# ================
# Run this AFTER creating the conda environment:
#
#   conda env create -f environment_fixed.yml --prefix /scratch/$USER/bulk_transformer_env
#   conda activate /scratch/$USER/bulk_transformer_env
#   bash install_torch.sh
#
# Why --prefix on /scratch?
#   DAIC's /home/nfs is an NFS mount that blocks certain file operations
#   (notably writing shared libraries like triton's libproton.so).
#   /scratch is local disk and has no such restrictions.

set -e

TORCH_VERSION="2.5.1"
CUDA_TAG="cu118"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "=============================================="
echo " Step 1: Pinning triton to 3.1.0"
echo " (avoids libproton.so NFS permissions error)"
echo "=============================================="
# Uninstall any triton that conda/pip may have pulled in (e.g. 3.2.0 from torch deps)
pip uninstall -y triton 2>/dev/null || true
pip install triton==3.1.0 --no-deps

echo ""
echo "=============================================="
echo " Step 2: Installing PyTorch ${TORCH_VERSION}+${CUDA_TAG}"
echo "=============================================="
pip install \
    torch==${TORCH_VERSION}+${CUDA_TAG} \
    torchvision==0.20.1+${CUDA_TAG} \
    torchaudio==2.5.1+${CUDA_TAG} \
    --index-url ${TORCH_INDEX} \
    --no-deps

# Install torch runtime deps that aren't already present
pip install \
    filelock fsspec jinja2 networkx sympy typing-extensions \
    --quiet

echo ""
echo "=============================================="
echo " Step 3: Installing torch-geometric sparse backends"
echo "=============================================="
pip install \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-cluster==1.6.3 \
    torch-spline-conv==1.2.2 \
    --find-links ${PYG_INDEX} \
    --no-deps

pip install torch-geometric==2.6.1 --no-deps

echo ""
echo "=============================================="
echo " Step 4: Install performer-pytorch without re-pulling torch"
echo "=============================================="
pip install performer-pytorch==1.1.4 --no-deps

echo ""
echo "=============================================="
echo " Verifying installation"
echo "=============================================="
python - <<'EOF'
import torch
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import GCNConv
from performer_pytorch import Performer

print(f"  torch          : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device    : {torch.cuda.get_device_name(0)}")
print(f"  torch_sparse   : OK")
print(f"  torch_geometric: OK")
print(f"  performer      : OK")
print("")
print("  All imports successful — environment is ready.")
EOF