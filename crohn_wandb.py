#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from pathlib import Path
import wandb
import prefect
from prefect import task, flow
import scanpy as sc
import anndata
import scgpt as scg
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Tasks
# -------------------------------------------------------

@task
def train_task(dataset_path: str, model_path: str, batch_size: int):
    """Train by embedding the dataset using scGPT."""
    logger = prefect.get_run_logger()

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    disease_adata = sc.read_h5ad(dataset_path)

    if "tissue" not in disease_adata.obs.columns:
        raise KeyError("Column 'tissue' not found in the dataset.")

    # Subset for colon tissue
    colon_index = disease_adata.obs.reset_index().query("tissue=='colon'").index.to_list()
    colon_adata = disease_adata[colon_index, :]

    # Convert to dense
    colon_adata = anndata.AnnData(
        X=colon_adata.to_df().values,
        obs=colon_adata.obs.copy(),
        var=colon_adata.var.copy()
    )

    if "disease" not in colon_adata.obs.columns:
        raise KeyError("Expected column 'disease' not found in AnnData object.")

    logger.info("Embedding data using scGPT...")
    ref_embed_adata = scg.tasks.embed_data(
        colon_adata,
        model_dir=Path(model_path),
        gene_col='feature_name',
        obs_to_save=list(colon_adata.obs.columns),
        batch_size=batch_size,
        return_new_adata=True,
    )

    # Log embedding shape
    wandb.log({"Embedding Shape": ref_embed_adata.X.shape})

    # Create table for wandb
    data = [list(emb) + [ct] for emb, ct in zip(ref_embed_adata.X, colon_adata.obs["disease"])]
    columns = [f"Embedding_{i}" for i in range(ref_embed_adata.X.shape[1])] + ["Cell Type"]
    embeddings_table = wandb.Table(data=data, columns=columns)
    wandb.log({"Embedding Table": embeddings_table})

    return ref_embed_adata

@task
def plot_task(ref_embed_adata):
    """Plot UMAP from the embeddings and log it."""
    logger = prefect.get_run_logger()

    logger.info("Running Scanpy UMAP...")
    sc.pp.neighbors(ref_embed_adata, use_rep="X")
    sc.tl.umap(ref_embed_adata)

    # Plot
    sc.pl.umap(ref_embed_adata, color='disease', frameon=False, wspace=0.4, show=False)
    wandb.log({"Scanpy UMAP Plot": wandb.Image(plt.gcf())})
    plt.close()

    # Log UMAP coordinates
    umap_data = [
        [float(coord[0]), float(coord[1]), cell_type]
        for coord, cell_type in zip(ref_embed_adata.obsm['X_umap'], ref_embed_adata.obs['disease'])
    ]
    umap_table = wandb.Table(columns=["UMAP1", "UMAP2", "Cell Type"], data=umap_data)
    wandb.log({"UMAP Coordinates Table": umap_table})

    logger.info("UMAP plotting completed.")


# Optional separate UMAP using raw embeddings
def custom_plot_umap(positive_class, negative_class, cell_types, emb, desc):
    reducer = umap.UMAP()
    embedding_umap = reducer.fit_transform(emb)
    color_map = {positive_class: 'red', negative_class: 'blue'}
    colors = [color_map.get(ct, 'grey') for ct in cell_types]

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=colors, s=10, alpha=0.8)
    plt.title(desc)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=positive_class, markersize=10, markerfacecolor='red'),
        Line2D([0], [0], marker='o', color='w', label=negative_class, markersize=10, markerfacecolor='blue'),
        Line2D([0], [0], marker='o', color='w', label='Other', markersize=10, markerfacecolor='grey')
    ]
    plt.legend(handles=legend_elements)

    wandb.log({f"Custom UMAP Plot - {desc}": wandb.Image(plt.gcf())})
    plt.close()

# -------------------------------------------------------
# Flow
# -------------------------------------------------------

@flow(name="scGPT Training and UMAP Flow")
def main_flow(dataset_path: str, model_path: str, batch_size: int = 128, train: bool = True, plot: bool = True):
    """Flow to train the model and plot UMAP."""
    try:
        wandb.init(
            project="scGPT Training",
            settings=wandb.Settings(start_method="thread"),
            config={
                "max_length": 1800,
                "batch_size": batch_size,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "dataset": "colon",
                "model_path": model_path
            }
        )
    except wandb.errors.UsageError as e:
        print(f"wandb.init failed: {e}")
    if train:
        ref_embed_adata = train_task(dataset_path, model_path, batch_size).result()
    if plot:
        plot_task(ref_embed_adata)

    wandb.finish()

# -------------------------------------------------------
# CLI
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scGPT training and UMAP visualization as a Prefect Flow.")
    parser.add_argument("--dataset_path", type=str, default="data/colon_adata.h5ad", help="Path to the dataset.")
    parser.add_argument("--model_path", type=str, default="models/human_model.pt", help="Path to the scGPT model.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding.")
    parser.add_argument("--train", action="store_true", help="Whether to train the model.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot UMAP.")
    args = parser.parse_args()

    # Call the flow
    main_flow(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        train=args.train,
        plot=args.plot
    )
