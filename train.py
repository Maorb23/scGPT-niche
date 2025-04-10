#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import argparse
import os
from pathlib import Path
import scanpy as sc
import anndata
import scgpt as scg
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class scGPT_niche:
    def __init__(self, colon_data_path, model_path, batch_size):
        self.colon_data_path = colon_data_path
        self.model_path = model_path
        self.batch_size = batch_size

    def embed(self):
        """Train by embedding the dataset using scGPT."""
        colon_adata = sc.read_h5ad(self.colon_data_path)
        logger.warning("Loaded colon dataset...")
        logger.warning(f"Shape of colon dataset: {colon_adata.shape}")

        logger.warning("Embedding data using scGPT...")
        ref_embed_adata = scg.tasks.embed_data(
            colon_adata,
            model_dir=Path(self.model_path),
            gene_col='feature_name',
            obs_to_save=list(colon_adata.obs.columns),
            batch_size=self.batch_size,
            return_new_adata=True,
        )
        logger.warning("Embedding done...")
        logger.warning(f"Shape of embedded data: {ref_embed_adata.shape}")

        return ref_embed_adata
    
    def fine_tune(self, ref_embed_adata):
        """Fine-tune the model using the embedded data."""
        # Fine-tuning logic goes here
        pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scGPT model.")
    parser.add_argument("--colon_data_path", default="ydata/data/processed/colon_adata.h5ad", help="Path to the colon data.")
    parser.add_argument("--model_path", default="ydata/models/best_model", help="Path to the scGPT model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    args = parser.parse_args()

    scgpt_niche = scGPT_niche(args.colon_data_path, args.model_path, args.batch_size)
    ref_embed_adata = scgpt_niche.embed()


