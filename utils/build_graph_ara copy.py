# build_ath_graph.py
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

EXPR_PATH  = '/home/alex/Documents/GitHub/Dataset_fusion_Microarray/new_storage/final_data/RMA_Microarray_Combined.csv'
GENE_INFO  = 'metadata/arabidopsis_gene_info.csv'
CHUNK_SIZE = 500    # genes per chunk — reduce to 200 if RAM is tight
TOP_K      = 20     # top edges per gene
PCC_THRESH = 0.4    # minimum |PCC|

gene_info = pd.read_csv(GENE_INFO)
gene_list = gene_info['tair_id'].tolist()

expr = pd.read_csv(EXPR_PATH, index_col=0).T
expr = expr[[c for c in gene_list if c in expr.columns]]
print(f'Expression matrix: {expr.shape}  (samples × genes)')

# Pre-standardise once — avoids repeating this inside each chunk
X = expr.values.astype(np.float32)           # [samples, genes]
X = X - X.mean(axis=0, keepdims=True)
std = X.std(axis=0, keepdims=True)
std[std == 0] = 1.0
X = X / std                                  # z-scored, [S, G]
n_samples, G = X.shape
print(f'Computing chunked PCC for {G} genes...')

rows, cols, vals = [], [], []

for i_start in tqdm(range(0, G, CHUNK_SIZE)):
    i_end  = min(i_start + CHUNK_SIZE, G)
    chunk  = X[:, i_start:i_end]             # [S, chunk]

    # PCC between chunk genes and ALL genes: [chunk, G]
    pcc_block = (chunk.T @ X) / n_samples    # pearson on z-scored data

    for local_i, global_i in enumerate(range(i_start, i_end)):
        row_pcc = pcc_block[local_i].copy()  # [G]
        row_abs = np.abs(row_pcc)
        row_abs[global_i] = 0                # exclude self

        # Top-K above threshold
        top_idx = np.argsort(row_abs)[-TOP_K:]
        for j in top_idx:
            if row_abs[j] >= PCC_THRESH:
                rows.append(global_i)
                cols.append(int(j))
                vals.append(float(row_pcc[j]))

edge_index  = torch.tensor([rows, cols], dtype=torch.long)
edge_weight = torch.tensor(vals,         dtype=torch.float32)

torch.save(edge_index,  'data/G_ath.pt')
torch.save(edge_weight, 'data/G_ath_weight.pt')
print(f'Done — {len(vals)} edges across {G} genes')
print(f'Avg edges per gene: {len(vals)/G:.1f}')