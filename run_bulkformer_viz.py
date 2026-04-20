import os
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
from collections import OrderedDict

# SparseTensor MUST come from torch_sparse directly.
# torch_geometric.typing re-exports it but silently falls back to None
# if the CUDA .so fails to load, causing the 3D-tensor GCN error.
from torch_sparse import SparseTensor

from utils.BulkFormer import BulkFormer
from model.config import model_params

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
ORGANISM   = 'arabidopsis'
EXPR_PATH  = '/home/alex/Documents/GitHub/Dataset_fusion_Microarray/new_storage/final_data/imputed.csv'
BATCH_SIZE = 8
N_SAMPLES  = 300   # set to None to use all samples

FILES = {
    'model_weights': 'model/checkpoints_ath/BulkFormer_ath_best.pt',
    'graph_ei':      'data/G_ath.pt',
    'graph_w':       'data/G_ath_weight.pt',
    'gene_info':     'metadata/arabidopsis_gene_info.csv',
}

matplotlib.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#0f1117',
    'axes.edgecolor':   '#3a3a4a', 'axes.labelcolor': '#c8c8d8',
    'xtick.color':      '#7a7a9a', 'ytick.color':     '#7a7a9a',
    'text.color':       '#c8c8d8', 'grid.color':       '#2a2a3a',
    'grid.linewidth':   0.5,       'font.family':     'monospace',
})


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── 1. Gene vocabulary ────────────────────────────────────────────────────
    gene_info = pd.read_csv(FILES['gene_info'])
    id_col    = 'tair_id' if 'tair_id' in gene_info.columns else gene_info.columns[0]
    all_genes = gene_info[id_col].drop_duplicates().tolist()

    # ── 2. Expression matrix ──────────────────────────────────────────────────
    print(f'Loading expression matrix from {EXPR_PATH}...')
    if N_SAMPLES is not None:
        # imputed.csv is genes × samples, so columns 1..N_SAMPLES+1 = first N samples
        print(f'Loading first {N_SAMPLES} samples (memory-saving mode)...')
        expr_df = pd.read_csv(EXPR_PATH, index_col=0,
                              usecols=list(range(N_SAMPLES + 1)))
    else:
        expr_df = pd.read_csv(EXPR_PATH, index_col=0)

    # Ensure orientation is samples × genes
    # if expr_df.shape[1] > expr_df.shape[0]:
    expr_df = expr_df.T
    print(f'Expression matrix: {expr_df.shape}  (samples × genes)')

    # ── 3. Sync vocabulary ────────────────────────────────────────────────────
    genes_in_expr = set(expr_df.columns)
    gene_list     = [g for g in all_genes if g in genes_in_expr]
    gene_idx      = {g: i for i, g in enumerate(gene_list)}
    GENE_LENGTH   = len(gene_list)
    print(f'Gene vocabulary synced: {GENE_LENGTH} genes')

    # Report any genes excluded (organellar, transposons, unannotated loci)
    excluded = sorted(genes_in_expr - set(gene_list))
    if excluded:
        print(f'Genes excluded from vocab (not in gene_info): {len(excluded)}'
              f'  e.g. {excluded[:4]}')

    # Align to vocab order, pad absent genes with -10
    expr_df  = expr_df[[c for c in gene_list if c in expr_df.columns]]
    missing  = [g for g in gene_list if g not in expr_df.columns]
    if missing:
        pad     = pd.DataFrame(-10.0, index=expr_df.index, columns=missing)
        expr_df = pd.concat([expr_df, pad], axis=1)
    input_df = expr_df[gene_list]
    expr_arr = input_df.values[:N_SAMPLES].astype(np.float32)
    print(f'Input array: {expr_arr.shape}')

    # ── 4. Load graph as SparseTensor (torch_sparse) ──────────────────────────
    # SparseTensor is what BulkFormer_block.py passes to the GCN layer.
    # Using torch.sparse_coo_tensor instead causes a 3D-tensor error because
    # PyG's GCN propagate() cannot handle the batched (3D) matmul.
    graph_ei  = torch.load(FILES['graph_ei'], map_location='cpu', weights_only=False)
    graph_w   = torch.load(FILES['graph_w'],  map_location='cpu', weights_only=False)
    graph_cpu = SparseTensor(
        row=graph_ei[1], col=graph_ei[0],
        value=graph_w,
        sparse_sizes=(GENE_LENGTH, GENE_LENGTH)
    )
    print(f'Graph loaded: {graph_ei.shape[1]} edges')

    # ── 5. Build model ────────────────────────────────────────────────────────
    model_params['graph']       = graph_cpu   # stays on CPU; moved to device below
    model_params['gene_emb']    = None
    model_params['gene_length'] = GENE_LENGTH
    model_params['dim']         = 128

    model = BulkFormer(**model_params).to(device)

    ckpt     = torch.load(FILES['model_weights'], map_location='cpu', weights_only=False)
    sd       = OrderedDict((k[7:] if k.startswith('module.') else k, v)
                           for k, v in ckpt.items())
    model_sd = model.state_dict()
    to_load  = {k: v for k, v in sd.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    skipped  = [k for k in sd if k not in to_load]
    model.load_state_dict(to_load, strict=False)
    if skipped:
        print(f'Weights: {len(to_load)} loaded, {len(skipped)} skipped: {skipped}')
    else:
        print(f'Weights: all {len(to_load)} keys matched')
    model.eval()

    # Move graph to device ONCE before inference (not inside each batch)
    model.graph = model.graph.to(device)

    # ── 6. Inference ──────────────────────────────────────────────────────────
    def run_model(arr):
        results = []
        with torch.no_grad():
            for i in tqdm(range(0, len(arr), BATCH_SIZE), desc='Extracting embeddings'):
                batch = torch.tensor(arr[i:i+BATCH_SIZE],
                                     dtype=torch.float32).to(device)
                out = model(batch, mask_prob=0.0, output_expr=False)
                results.append(out.cpu())
        return torch.cat(results, dim=0)

    print('Running BulkFormer...')
    embeddings = run_model(expr_arr)   # [Samples, Genes, Dim]
    print(f'Embeddings shape: {embeddings.shape}')

    os.makedirs('figures',    exist_ok=True)
    os.makedirs('embeddings', exist_ok=True)

    # ── 7. Sample UMAP ────────────────────────────────────────────────────────
    print('Computing sample UMAP...')
    sample_emb  = embeddings.mean(dim=1).numpy()   # [Samples, Dim]
    sample_umap = umap.UMAP(n_neighbors=15, min_dist=0.1,
                            random_state=42).fit_transform(sample_emb)

    plt.figure(figsize=(10, 8))
    plt.scatter(sample_umap[:, 0], sample_umap[:, 1],
                s=10, alpha=0.7, color='#00e5c0')
    plt.title(f'BulkFormer Sample UMAP  (n={len(expr_arr)})')
    plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(f'figures/umap_samples_{ORGANISM}.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print('Saved figures/umap_samples_arabidopsis.png')

    # ── 8. Gene UMAP ──────────────────────────────────────────────────────────
    print('Computing gene UMAP...')
    gene_emb  = embeddings.mean(dim=0).numpy()     # [Genes, Dim]
    gene_umap = umap.UMAP(n_neighbors=15, min_dist=0.1,
                           random_state=42).fit_transform(gene_emb)

    plt.figure(figsize=(10, 8))
    plt.scatter(gene_umap[:, 0], gene_umap[:, 1],
                s=2, alpha=0.5, color='#ff6b6b')
    plt.title(f'BulkFormer Gene UMAP  (n={GENE_LENGTH})')
    plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(f'figures/umap_genes_{ORGANISM}.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print('Saved figures/umap_genes_arabidopsis.png')

    # ── 9. Save embeddings ────────────────────────────────────────────────────
    np.save(f'embeddings/{ORGANISM}_sample_embeddings.npy', sample_emb)
    np.save(f'embeddings/{ORGANISM}_gene_embeddings.npy',   gene_emb)
    print(f'\nDone. Figures in figures/  |  Embeddings in embeddings/')


if __name__ == '__main__':
    main()
