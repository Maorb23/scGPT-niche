import torch
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
import sys
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class scGPT_niche:
    def __init__(self, colon_data_path, model_path,model_path_spatial,
     batch_size, fine_tune=False):
        self.colon_data_path = colon_data_path
        self.model_path = model_path
        self.model_path_spatial = model_path_spatial
        self.batch_size = batch_size
    
    def embed_niche(self):
    
        """
        Extract embeddings from a pre-trained Nicheformer model and store them in AnnData.

        Args:
            config (dict): Configuration dictionary containing all necessary parameters
        """
        import os
        import numpy as np
        import pytorch_lightning as pl
        import torch
        from torch.utils.data import DataLoader
        import anndata as ad
        from typing import Optional, Dict, Any
        import argparse
        import yaml

        from nicheformer.models import Nicheformer
        from nicheformer.data import dataset as NicheformerDataset
        config = {
        'technology_mean_path': 'data/model_means/merfish_mean_script.npy', #'path/to/technology_mean.npy',  # Path to technology mean file
        #'checkpoint_path': '/lustre/groups/ml01/projects/2023_nicheformer/pretrained_models/everything_heads_16_blocks_12_maxsteps_30661140_FINAL/epoch=1-step=265000.ckpt',  # Path to model checkpoint
        'batch_size': 32,
        'max_seq_len': 1500, 
        'aux_tokens': 30, 
        'chunk_size': 1000, # to prevent OOM
        'num_workers': 4,
        'precision': 32,
        'embedding_layer': -1,  # Which layer to extract embeddings from (-1 for last layer)
        'embedding_name': 'embeddings'  # Name suffix for the embeddxing key in adata.obsm
        }
        
        # Set random seed for reproducibility
        pl.seed_everything(42)

        # Load data
        adata = ad.read_h5ad(self.colon_data_path)
        adata.obs['modality'] = 4 # Spatial, Does it mean that it uses spatial context or does it mean That I need to add a spatial context?
        adata.obs['specie'] = 5 # human
        adata.obs['assay'] = 7 #Merfish
        technology_mean = np.load(config['technology_mean_path'])

        # Create dataset for all cells (no train/val/test split needed)
        dataset = NicheformerDataset(
            adata=adata,
            technology_mean=technology_mean,
            split=None,  # Use all cells
            max_seq_len=config.get('max_seq_len', 4096),
            aux_tokens=config.get('aux_tokens', 30),
            chunk_size=config.get('chunk_size', 1000)
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )

        # Load pre-trained model
        model = ad.read_h5ad('data/model_means/model.h5ad')
        model.eval()  # Set to evaluation mode

        # Configure trainer
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            default_root_dir=config['output_dir'],
            precision=config.get('precision', 32),
        )

        # Get embeddings
        print("Extracting embeddings...")
        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Get embeddings from the model
                emb = model.get_embeddings(
                    batch=batch,
                    layer=config.get('embedding_layer', -1)  # Default to last layer
                )
                embeddings.append(emb.cpu().numpy())

        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings, axis=0)

        # Store embeddings in AnnData object
        #embedding_key = f"X_niche_{config.get('embedding_name', 'embeddings')}"
        #adata.obsm[embedding_key] = embeddings

        # Save updated AnnData
        #adata.write_h5ad(config['output_path'])

        #print(f"Embeddings saved to {config['output_path']} in obsm['{embedding_key}']")
        logger.warning("Embedding done...")
        logger.warning(f"Shape of embedded data: {embeddings.shape}")
        logger.warning("Embedding done...")
        logger.warning(f"Shape of embedded data: {embeddings.shape}")
    
    def fine_tune(self, ref_embed_adata, epochs=25, optimizer_type="adam"):
        train_losses = []
        test_loss_list = []
        emb = ref_embed_adata.X  # shape (n_cells, n_genes)
        # Instead of nn.Linear(512, 2)
        linear_probe = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 3)
        )

        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(linear_probe.parameters(), lr=1e-3, weight_decay=1e-4)
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(linear_probe.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        # Extract embeddings and labels
        emb = ref_embed_adata.X  # shape (n_cells, n_genes)
        region_labels = ref_embed_adata.obs['region'].values

        # Map labels to integers
        label_mapping = {"Stroma": 0, "Tumor": 1, "TLS": 2}
        labels = np.array([label_mapping[r] for r in region_labels])

        logger.warning(f"Embedding shape: {emb.shape}")
        logger.warning(f"Labels distribution: {np.bincount(labels)}")

        # 1. Create the X and y tensors
        X_tensor = torch.tensor(emb, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        # 2. Shuffle
        perm = torch.randperm(X_tensor.size(0))
        X_tensor = X_tensor[perm]
        y_tensor = y_tensor[perm]

        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # Train/test split (80/20)
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_test = n_total - n_train
        X_train, X_test, y_train, y_test = train_test_split(
        X_tensor.numpy(), y_tensor.numpy(),  # Convert to numpy for sklearn
        test_size=0.2,
        random_state=42,
        stratify=y_tensor.numpy()
        )

        # 3. Wrap back into TensorDatasets
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

        # 4. Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(linear_probe.parameters(), lr=1e-3, weight_decay=1e-4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        linear_probe.to(device)

        logger.warning("Starting training...")

        # Training loop
        linear_probe.train()
        all_preds_train = []
        all_targets_train = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                logits = linear_probe(batch_X)
                preds = torch.argmax(logits, dim=1)
                all_targets_train.append(batch_y.cpu().numpy())
                all_preds_train.append(preds.cpu().numpy())
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                train_losses.append(epoch_loss)
                
            epoch_loss /= len(train_loader)
            logger.warning(f"Epoch {epoch+1}/{epochs}, Mean Train Loss: {epoch_loss:.4f}")
        

            # Evaluation
            linear_probe.eval()
            all_preds_test = []
            all_targets_test = []
            epoch_test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    logits = linear_probe(batch_X)
                    preds = torch.argmax(logits, dim=1)
                    test_loss = criterion(logits, batch_y)
                    epoch_test_loss += test_loss
                    all_preds_test.append(preds.cpu().numpy())
                    all_targets_test.append(batch_y.cpu().numpy())
                    test_loss_list.append(test_loss)
                epoch_test_loss /= len(test_loader)
                logger.warning(f"Epoch {epoch+1}/{epochs}, Mean Test Loss: {epoch_test_loss:.4f}")
        all_preds_train = np.concatenate(all_preds_train)
        all_targets_train = np.concatenate(all_targets_train)
        all_preds_test = np.concatenate(all_preds_test)
        all_targets_test = np.concatenate(all_targets_test)

        f1_macro_train = f1_score(all_targets_train, all_preds_train, average='macro')
        f1_micro_train = f1_score(all_targets_train, all_preds_train, average='micro')  

        logger.warning(f"F1 macro train score: {f1_macro_train}, F1 micro train score: {f1_micro_train}")

        logger.warning("Training done. Evaluating...")

        f1_macro_test = f1_score(all_targets_test, all_preds_test, average='macro')
        f1_micro_test = f1_score(all_targets_test, all_preds_test, average='micro')

        logger.warning(f"F1 Macro Test: {f1_macro_test:.4f}")
        logger.warning(f"F1 Micro Test: {f1_micro_test:.4f}")
        print("Classification report Train:")
        print(classification_report(all_targets_train, all_preds_train))
        print("Classification report Test:")
        print(classification_report(all_targets_test, all_preds_test))
        logger.warning("Evaluation done.")
        return linear_probe, train_losses, test_loss_list, f1_macro_train, f1_micro_train, f1_macro_test, f1_micro_test
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scGPT model.")
    parser.add_argument("--colon_data_path", default="ydata/data/processed/Vizgen-hCRC-1313910_VS39_colon.h5ad", help="Path to the colon data.")
    parser.add_argument("--model_path", default="ydata/models/best_model", help="Path to the scGPT model.")
    parser.add_argument("--model_path_spatial",  default="ydata/models/best_model_spatial", help="Path to the scGPT_spatial model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--fine_tune", default= False, action="store_true", help="Fine-tune the model.")
    parser.add_argument("--embed", default=False, action="store_true", help="embed the model.")
    parser.add_argument("--embed_spatial", default=False, action="store_true", help="embed the model.")
    parser.add_argument("--embed_niche", default=False, action="store_true", help="embed the model.")
    args = parser.parse_args()

    scgpt_niche = scGPT_niche(args.colon_data_path, args.model_path, args.model_path_spatial, args.batch_size, args.fine_tune)
    if args.embed:
        ref_embed_adata = scgpt_niche.embed()
    if args.embed_spatial:
        ref_embed_adata =scgpt_niche.embed_spatial()
    if args.embed_niche:
        ref_embed_adata = scgpt_niche.embed_niche()
    if args.fine_tune:
        linear_probe, train_losses, test_loss_list, f1_macro_train, f1_micro_train, f1_macro_test, f1_micro_test = scgpt_niche.fine_tune(ref_embed_adata)
        logger.warning(f"F1 Macro: {f1_macro_test:.4f}")
        logger.warning(f"F1 Micro: {f1_micro_test:.4f}")
    else:
        logger.warning("Fine-tuning not selected. Exiting...")

