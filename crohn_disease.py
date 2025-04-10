#!/usr/bin/env python
# coding: utf-8
import argparse
import os
from pathlib import Path
import wandb
import scanpy as sc
import anndata
import scgpt as scg
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Initialize wandb with sensible defaults.
# In production, you might want to replace these hardcoded values with command-line args or configs.
hyperparameter_defaults = {
    "max_length": 1800,
    "batch_size": 128,
    "device": "cuda",
    "dataset": "colon",
    "model_path": "Your_Model_Path",  # <-- Update this path!
}
wandb.init(
    project="scGPT Training",
    settings=wandb.Settings(start_method="thread"),
    config=hyperparameter_defaults
)

max_length = wandb.config.max_length
batch_size = wandb.config.batch_size
device = wandb.config.device
model_path = wandb.config.model_path

# Paths
disease_dataset_path =  disease_dataset_path

def plot_umap(positive_class, negative_class, cell_types, emb, desc):
    """Generate and log a UMAP plot from an embedding array."""
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

    # Log the current figure properly by capturing the current figure instance.
    wandb.log({f"UMAP Plot - {desc}": wandb.Image(plt.gcf())})
    plt.close()


# Load dataset and check that the file exists.
if not os.path.exists(disease_dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {disease_dataset_path}")

disease_adata = sc.read_h5ad(disease_dataset_path)

# Check the necessary columns
if "tissue" not in disease_adata.obs.columns:
    raise KeyError("Column 'tissue' not found in the dataset.")

# Subset for colon tissue
colon_index = disease_adata.obs.reset_index().query("tissue=='colon'").index.to_list()
colon_adata = disease_adata[colon_index, :]

# Convert to dense data if needed and rebuild AnnData to avoid referencing issues.
dense_X = colon_adata.to_df().values
colon_adata = anndata.AnnData(
    X=dense_X,
    obs=colon_adata.obs.copy(),
    var=colon_adata.var.copy(),
    obsm=colon_adata.obsm.copy() if hasattr(colon_adata, "obsm") else None,
    varm=colon_adata.varm.copy() if hasattr(colon_adata, "varm") else None,
    uns=colon_adata.uns.copy() if hasattr(colon_adata, "uns") else None,
    layers=colon_adata.layers.copy() if hasattr(colon_adata, "layers") else None
)

# Check that the expected metadata exists for logging (e.g., "disease").
if "disease" not in colon_adata.obs.columns:
    raise KeyError("Expected column 'disease' not found in the AnnData object.")

# Embed the data using scGPT
ref_embed_adata = scg.tasks.embed_data(
    colon_adata,
    model_dir=Path(model_path),
    gene_col='feature_name',
    obs_to_save=list(colon_adata.obs.columns),
    batch_size=batch_size,
    return_new_adata=True,
)

wandb.log({"Embedding Shape": ref_embed_adata.X.shape})

# Log the embedding table with clear column naming.
embeddings_table = wandb.Table(
    data=ref_embed_adata.X,
    columns=[f"Embedding_{i}" for i in range(ref_embed_adata.X.shape[1])]
)
# Add metadata for analysis (e.g., cell type information from the "disease" column)
embeddings_table.add_column("Cell Type", colon_adata.obs["disease"].tolist())
wandb.log({"Embedding Table": embeddings_table})

# UMAP visualization using Scanpy utilities
sc.pp.neighbors(ref_embed_adata, use_rep="X")
sc.tl.umap(ref_embed_adata)

# Generate and log a static UMAP figure.
umap_fig = sc.pl.umap(ref_embed_adata, color='disease', frameon=False, wspace=0.4, show=False)
# Instead of using fig.to_html() (which is not valid for matplotlib figures), log the image directly.
wandb.log({"UMAP Plot": wandb.Image(umap_fig)})
plt.close(umap_fig)

# Create and log a separate table for UMAP coordinates.
umap_table = wandb.Table(
    columns=["UMAP1", "UMAP2", "Cell Type"],
    data=[
        [float(coord[0]), float(coord[1]), cell_type]
        for coord, cell_type in zip(ref_embed_adata.obsm['X_umap'], ref_embed_adata.obs['disease'])
    ]
)
wandb.log({"UMAP Coordinates Table": umap_table})

# Optionally, you can also call your custom plot_umap if needed.
# For example:
# plot_umap(positive_class='disease_positive', negative_class='disease_negative',
#           cell_types=colon_adata.obs["disease"].tolist(), emb=ref_embed_adata.X, desc="Custom UMAP")

wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scGPT on a dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file.")
