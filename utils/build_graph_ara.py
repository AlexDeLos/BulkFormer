# build_ath_graph.py
import pandas as pd
import numpy as np
import torch

# EXPR_PATH  = '/home/alex/Documents/GitHub/Dataset_fusion_Microarray/new_storage/final_data/RMA_Microarray_Combined.csv'
EXPR_PATH  = '/tudelft.net/staff-umbrella/GeneExpressionStorage/final_data/imputed.csv'
GENE_INFO  = 'metadata/arabidopsis_gene_info.csv'

gene_info = pd.read_csv(GENE_INFO)
gene_list = gene_info['tair_id'].tolist()

expr = pd.read_csv(EXPR_PATH, index_col=0).T
# Keep only genes in vocabulary, transpose to samples x genes
expr = expr[[c for c in gene_list if c in expr.columns]]
print(f'Expression matrix: {expr.shape}')

# Pearson correlation — may need chunking for 27k genes
print('Computing correlations...')
corr = expr.corr(method='pearson')  # [G, G]

# Apply thresholds: |PCC| > 0.4, top-20 edges per gene
G = len(corr)
rows, cols, vals = [], [], []
corr_abs = corr.abs()

for i in range(G):
    col_vals = corr_abs.iloc[i].values.copy()
    col_vals[i] = 0  # exclude self
    top20_idx = np.argsort(col_vals)[-20:]
    for j in top20_idx:
        if col_vals[j] >= 0.4:
            rows.append(i)
            cols.append(j)
            vals.append(corr.iloc[i, j])

edge_index  = torch.tensor([rows, cols], dtype=torch.long)
edge_weight = torch.tensor(vals, dtype=torch.float32)

torch.save(edge_index,  '/tudelft.net/staff-umbrella/GeneExpressionStorage/final_data/graph_data/G_ath.pt')
torch.save(edge_weight, '/tudelft.net/staff-umbrella/GeneExpressionStorage/final_data/graph_data/G_ath_weight.pt')
print(f'Graph: {len(vals)} edges across {G} genes')