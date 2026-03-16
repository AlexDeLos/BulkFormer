# train_arabidopsis.py
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from collections import OrderedDict
from tqdm import tqdm
from torch_geometric.typing import SparseTensor
from utils.BulkFormer import BulkFormer

# ── Config ───────────────────────────────────────────────────────────────────
EXPR_PATH  = '/home/alex/Documents/GitHub/Dataset_fusion_Microarray/new_storage/final_data/RMA_Microarray_Combined.csv'
GENE_INFO   = 'metadata/arabidopsis_gene_info.csv'
GRAPH_PATH  = 'data/G_ath.pt'
WEIGHT_PATH = 'data/G_ath_weight.pt'
SAVE_DIR    = 'model/checkpoints_ath'
os.makedirs(SAVE_DIR, exist_ok=True)

DIM        = 128
GB_REPEAT  = 1
P_REPEAT   = 1
FULL_HEAD  = 8
MASK_RATIO = 0.15
BATCH_SIZE = 16
LR         = 1e-4
EPOCHS     = 50
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Gene vocab ───────────────────────────────────────────────────────────────
gene_info   = pd.read_csv(GENE_INFO)
gene_list   = gene_info['tair_id'].tolist()
GENE_LENGTH = len(gene_list)
gene_idx    = {g: i for i, g in enumerate(gene_list)}
print(f'Vocabulary: {GENE_LENGTH} genes')

# ── Expression data ──────────────────────────────────────────────────────────
expr_df = pd.read_csv(EXPR_PATH, index_col=0)
# Align to vocabulary
missing = list(set(gene_list) - set(expr_df.columns))
pad     = pd.DataFrame(0.0, index=expr_df.index, columns=missing)
expr_df = pd.concat([expr_df, pad], axis=1)[gene_list]
expr_np = expr_df.values.astype(np.float32)
print(f'Expression matrix: {expr_np.shape}')

# ── Graph ────────────────────────────────────────────────────────────────────
ei = torch.load(GRAPH_PATH,  weights_only=False)
ew = torch.load(WEIGHT_PATH, weights_only=False)
graph = SparseTensor(row=ei[1], col=ei[0], value=ew).to(DEVICE)

# ── Dataset ──────────────────────────────────────────────────────────────────
class ExprDataset(Dataset):
    def __init__(self, expr, mask_ratio=0.15):
        self.expr       = expr
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.expr)

    def __getitem__(self, idx):
        x    = self.expr[idx].copy()
        true = x.copy()
        # Mask random genes
        obs     = np.where(x != 0)[0]
        k       = max(1, int(len(obs) * self.mask_ratio))
        chosen  = np.random.choice(obs, size=k, replace=False)
        x[chosen] = -10.0
        mask    = np.zeros(len(x), dtype=np.float32)
        mask[chosen] = 1.0
        return (torch.tensor(x,    dtype=torch.float32),
                torch.tensor(true, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32))

dataset = ExprDataset(expr_np, mask_ratio=MASK_RATIO)
n_val   = max(1, int(0.1 * len(dataset)))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f'Train: {n_train}  Val: {n_val}')

# ── Model ────────────────────────────────────────────────────────────────────
model = BulkFormer(
    dim=DIM, graph=graph, gene_emb=None,
    gene_length=GENE_LENGTH,
    bin_head=12, full_head=FULL_HEAD,
    bins=0, gb_repeat=GB_REPEAT, p_repeat=P_REPEAT
).to(DEVICE)
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_dl), epochs=EPOCHS
)

# ── Training loop ────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    with torch.set_grad_enabled(train):
        for x, true, mask in tqdm(loader, leave=False):
            x, true, mask = x.to(DEVICE), true.to(DEVICE), mask.to(DEVICE)
            pred = model(x, mask_prob=MASK_RATIO, output_expr=True)
            # Loss only on masked positions
            loss = ((pred - true) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item()
    return total_loss / len(loader)

best_val = float('inf')
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_dl, train=True)
    val_loss   = run_epoch(val_dl,   train=False)
    print(f'Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}')

    ckpt_path = f'{SAVE_DIR}/BulkFormer_ath_epoch{epoch:02d}.pt'
    torch.save(model.state_dict(), ckpt_path)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), f'{SAVE_DIR}/BulkFormer_ath_best.pt')
        print(f'  → New best: {best_val:.4f}')