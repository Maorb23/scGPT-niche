#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
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
from preprocess import Preprocess
from train import scGPT_niche
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
from plots import Plots
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Tasks
# -------------------------------------------------------


@task
def preprocess_task(dataset_path: str, describe: bool = True):

    logger = prefect.get_run_logger()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    preprocessor = Preprocess(dataset_path, describe)
    colon_adata, full_describe, N_genes = preprocessor.preprocess_data()
    if describe:
        #full_describe_clean = full_describe.copy()
        #for col in full_describe_clean.columns:
            #if full_describe_clean[col].dtype == object:  # string columns
                #full_describe_clean[col] = full_describe_clean[col].fillna("N/A")
            #else:  # numeric columns
                #full_describe_clean[col] = full_describe_clean[col].fillna(-1)  # or 0, or keep as NaN
        logger.info("Plotting and logging cell distribution by sex...")

        # Create the plot
        sex_counts = colon_adata.obs['sex'].value_counts()
        plt.figure(figsize=(4, 4))
        sex_counts.plot(kind='bar')
        plt.title("Cell Distribution by Sex")
        plt.xlabel("Sex")
        plt.ylabel("Count")
        plt.tight_layout()

        # Log the plot to WandB
        wandb.log({"Sex Distribution Plot": wandb.Image(plt.gcf())}) # Load the plot to WandB
        plt.close()        
        N_genes_clean = N_genes.fillna("N/A")
        #full_table = wandb.Table(dataframe=full_describe_clean)
        #wandb.log({"Full Describe": full_table})
        # For N_genes
        genes_table = wandb.Table(dataframe=N_genes_clean)
        wandb.log({"Genes Count": genes_table})
    return colon_adata

@task 
def EDA_plots(colon_adata, dataset_path):
    logger = prefect.get_run_logger()
    plots = Plots(dataset_path)
    logger.info("Plotting and logging cell distribution by sex...")
    fig1 = plots.plot_most_expressed_genes(colon_adata, n_genes=20)
    wandb.log({"Most Expressed Genes": wandb.Image(fig1)})
    fig2 = plots.plot_combined_distributions(colon_adata, save_html=True)
    # Now load the saved HTML file for wandb
    wandb.log({"Combined Distributions": wandb.Html(open("interactive_distribution.html"))})
    return fig1,  fig2
    


"""
def train_task(dataset_path: str, model_path: str, batch_size: int):

    logger = prefect.get_run_logger()
    disease_adata = sc.read_h5ad(dataset_path)

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

    # Randomly select 10 indices
    num_samples = min(10, ref_embed_adata.shape[0])
    random_indices = np.random.choice(ref_embed_adata.shape[0], size=num_samples, replace=False)

    # Sampled data
    sampled_data = [
        list(ref_embed_adata.X[idx]) + [colon_adata.obs["disease"].iloc[idx]]
        for idx in random_indices
    ]
    columns = [f"Embedding_{i}" for i in range(ref_embed_adata.X.shape[1])] + ["Cell Type"]

    # Create a smaller WandB table
    sampled_table = wandb.Table(data=sampled_data, columns=columns)
    wandb.log({"Embedding Table (Sampled 10 Rows)": sampled_table})

    return ref_embed_adata
"""

@task
def train_task(colon_path: str, model_path: str, batch_size: int,fine_tune):

    trainer = scGPT_niche(colon_path, model_path, batch_size,fine_tune)
    colon_adata = sc.read_h5ad(colon_path)
    ref_embed_adata = trainer.embed()
    if fine_tune:
        linear_probe, train_losses, test_loss_list, f1_macro_train, f1_micro_train, f1_macro_test, f1_micro_test = trainer.fine_tune(ref_embed_adata)
    # Randomly select 10 indices
    num_samples = min(10, ref_embed_adata.shape[0])
    random_indices = np.random.choice(ref_embed_adata.shape[0], size=num_samples, replace=False)
    sampled_data = [
        list(ref_embed_adata.X[idx]) + [colon_adata.obs["disease"].iloc[idx]]
        for idx in random_indices
    ]
    columns = [f"Embedding_{i}" for i in range(ref_embed_adata.X.shape[1])] + ["Cell Type"]

    # Create a smaller WandB table
    sampled_table = wandb.Table(data=sampled_data, columns=columns)
    wandb.log({"Embedding Table (Sampled 10 Rows)": sampled_table})
    if fine_tune:
        return ref_embed_adata, linear_probe, train_losses, test_loss_list, f1_macro_train, f1_micro_train, f1_macro_test, f1_micro_test
    else:
        return ref_embed_adata

@task
def plot_task(ref_embed_adata):

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
def main_flow(dataset_path: str, colon_path, model_path: str, batch_size: int = 128, train: bool = False, plot: bool = False,
        preprocess: bool = False, custom_plot: bool = False, eda_plots: bool = False, fine_tune = False):
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
    if preprocess:
        colon_adata = preprocess_task(dataset_path)
    
    if eda_plots:
        EDA_plots(colon_adata, dataset_path)
    if train:
        if fine_tune:
            ref_embed_adata, linear_probe, train_losses, test_loss_list, f1_macro_train, f1_micro_train, f1_macro_test, f1_micro_test = train_task(colon_path, model_path, batch_size, fine_tune) 
        else:
            ref_embed_adata = train_task(colon_path, model_path, batch_size, fine_tune)
    if plot:
        plot_task(ref_embed_adata)
    if custom_plot:
        positive_class = "Crohn disease"
        negative_class = "normal"

        custom_plot_umap(positive_class, negative_class,
                          colon_adata.obs['disease'].tolist(), ref_embed_adata.X, "Crohn disease")

    wandb.finish()

# -------------------------------------------------------
# CLI
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scGPT training and UMAP visualization as a Prefect Flow.")
    parser.add_argument("--dataset_path", type=str, default="ydata/data/base_dataset.h5ad", help="Path to the dataset.")
    parser.add_argument("--colon_path", type=str, default="ydata/data/processed/colon_adata.h5ad", help="Path to the colon data.")
    parser.add_argument("--model_path", type=str, default="ydata/models/best_model", help="Path to the scGPT model.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding.")
    parser.add_argument("--train", action="store_true", help="Whether to train the model.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot UMAP.")
    parser.add_argument("--preprocess", action="store_true", help="Whether to preprocess the data.")
    parser.add_argument("--custom_plot", action="store_true", help="Whether to create a custom UMAP plot.")
    parser.add_argument("--eda_plots", action="store_true", help="Whether to create EDA plots.")
    parser.add_argument("--fine_tune", action="store_true", help="Whether to use fine-tune.")
    args = parser.parse_args()

    # Call the flow
    main_flow(
        dataset_path=args.dataset_path,
        colon_path=args.colon_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        train=args.train,
        plot=args.plot,
        preprocess=args.preprocess,
        custom_plot=args.custom_plot,
        eda_plots=args.eda_plots,
        fine_tune=args.fine_tune
    )

