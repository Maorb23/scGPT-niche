#!/usr/bin/env python
# coding: utf-8

import sys
from pathlib import Path
import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
import warnings
import torch

# Optional: Uncomment if you want evaluation metrics
# from scgpt_spatial.utils import eval_scib_metrics

# Add scgpt_spatial module to path
sys.path.insert(0, "../")
import scgpt_spatial

warnings.filterwarnings("ignore", category=ResourceWarning)

def run_embedding_and_plotting():
    # --- Step 1: Load dataset ---
    adata = sc.read_h5ad('ydata/data/processed_fetal_lung_visium_xenium.h5ad')
    print(f"Loaded AnnData: {adata.shape}")

    print("Visium cell types:", adata[adata.obs['batch_id'] == 'visium'].obs.celltype.unique())
    print("Xenium cell types:", adata[adata.obs['batch_id'] == 'xenium'].obs.celltype.unique())

    # --- Step 2: Run zero-shot embedding ---
    model_dir = "ydata/models/best_model_spatial"
    gene_col = 'feature_name'
    cell_type_col = 'celltype'
    batch_id_col = 'batch_id'

    ref_embed_adata = scgpt_spatial.tasks.embed_data(
        adata,
        model_dir,
        gene_col=gene_col,
        obs_to_save=cell_type_col,
        batch_size=64,
        return_new_adata=True,
    )

    ref_embed_adata.obsm['X'] = ref_embed_adata.X.copy()
    ref_embed_adata.obs['batch_id'] = adata.obs['batch_id']

    # --- Optional Step: Eval ---
    # result_dict = eval_scib_metrics(ref_embed_adata, batch_key=batch_id_col, label_key=cell_type_col)
    # print(result_dict)

    # --- Step 3: UMAP and Plots ---
    sc.pp.neighbors(ref_embed_adata, use_rep="X")
    sc.tl.umap(ref_embed_adata)

    custom_palette = ['#23b3b3', '#ff8a5b', '#6aa84f', '#8e44ad', '#f1c40f']
    sc.pl.umap(ref_embed_adata, color=cell_type_col, palette=custom_palette, frameon=False, show=False, wspace=0.4)
    plt.savefig("umap_cell_labels.png", dpi=300, bbox_inches="tight")

    custom_palette = ['#4a6fa5', '#ea9999', '#9b6bd3', '#52c3a3', '#d8c656', '#6aa84f']
    sc.pl.umap(ref_embed_adata, color='batch_id', palette=custom_palette, frameon=False, show=False, wspace=0.4)
    plt.savefig("umap_batch_labels.png", dpi=300, bbox_inches="tight")

    print("✅ Embedding + UMAP complete. Images saved.")

# --- Main Entry ---
if __name__ == "__main__":
    run_embedding_and_plotting()
