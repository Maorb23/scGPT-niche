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
from torch import nn   
import torch.optim as optim
import umap
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class scGPT_niche:
    def __init__(self, colon_data_path, model_path, batch_size, fine_tune=False):
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
    
    def fine_tune(self, ref_embed_adata, epochs=10):
        train_losses = []
        test_loss_list = []
        # Extract embeddings and labels
        emb = ref_embed_adata.X  # shape (n_cells, n_genes)
        disease_labels = ref_embed_adata.obs['disease'].values

        # Map labels to integers
        label_mapping = {"normal": 0, "Crohn disease": 1}
        labels = np.array([label_mapping[d] for d in disease_labels])

        logger.warning(f"Embedding shape: {emb.shape}")
        logger.warning(f"Labels distribution: {np.bincount(labels)}")

        # Convert to torch tensors
        X_tensor = torch.tensor(emb, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # Train/test split (80/20)
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_test = n_total - n_train
        train_dataset, test_dataset = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Define linear probe
        linear_probe = nn.Linear(emb.shape[1], 2)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(linear_probe.parameters(), lr=1e-3, weight_decay=1e-4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        linear_probe.to(device)

        logger.warning("Starting training...")

        # Training loop
        linear_probe.train()
        all_preds = []
        all_targets = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                logits = linear_probe(batch_X)
                preds = torch.argmax(logits, dim=1)
                all_targets.append(batch_y.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                train_losses.append(epoch_loss)
                logger.warning(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        f1_macro_train = f1_score(all_targets, all_preds, average='macro')
        f1_micro_train = f1_score(all_targets, all_preds, average='micro')  

        logger.warning(f"F1 macro score: {f1_macro_train}, F1 micro score: {f1_micro_train}")

        logger.warning("Training done. Evaluating...")

        # Evaluation
        linear_probe.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = linear_probe(batch_X)
                preds = torch.argmax(logits, dim=1)
                test_loss = criterion(logits, batch_y)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                test_loss_list.append(test_loss)

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        f1_macro_test = f1_score(all_targets, all_preds, average='macro')
        f1_micro_test = f1_score(all_targets, all_preds, average='micro')

        logger.warning(f"F1 Macro: {f1_macro_test:.4f}")
        logger.warning(f"F1 Micro: {f1_micro_test:.4f}")

        return linear_probe, train_losses, test_loss_list, f1_macro_train, f1_micro_train, f1_macro_test, f1_micro_test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scGPT model.")
    parser.add_argument("--colon_data_path", default="ydata/data/processed/colon_adata.h5ad", help="Path to the colon data.")
    parser.add_argument("--model_path", default="ydata/models/best_model", help="Path to the scGPT model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--fine_tune", default= False, action="store_true", help="Fine-tune the model.")
    args = parser.parse_args()

    scgpt_niche = scGPT_niche(args.colon_data_path, args.model_path, args.batch_size)
    ref_embed_adata = scgpt_niche.embed()
    if args.fine_tune:
        linear_probe, train_losses, test_loss_list, f1_macro_train, f1_micro_train, f1_macro_test, f1_micro_test = scgpt_niche.fine_tune(ref_embed_adata)
        logger.warning(f"F1 Macro: {f1_macro_test:.4f}")
        logger.warning(f"F1 Micro: {f1_micro_test:.4f}")
    else:
        logger.warning("Fine-tuning not selected. Exiting...")


