#!/usr/bin/env python
# coding: utf-8

# # Extract Embeddings from Pre-trained Nicheformer Model
# 
# This notebook extracts embeddings from a pre-trained Nicheformer model and stores them in an AnnData object.

# In[1]:


import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import anndata as ad
from typing import Optional, Dict, Any
from tqdm import tqdm

from nicheformer.models import Nicheformer
from nicheformer.data import NicheformerDataset


# ## Configuration
# 
# Set up the configuration parameters for the embedding extraction.

# In[2]:


config = {
    'data_path': '/lustre/groups/ml01/projects/2023_nicheformer_data_anna.schaar/spatial/preprocessed/human/nanostring_cosmx_human_liver.h5ad', #'path/to/your/data.h5ad',  # Path to your AnnData file
    'technology_mean_path': '/lustre/groups/ml01/projects/2023_nicheformer/data/data_to_tokenize/cosmx_mean_script.npy', #'path/to/technology_mean.npy',  # Path to technology mean file
    'checkpoint_path': '/lustre/groups/ml01/projects/2023_nicheformer/pretrained_models/everything_heads_16_blocks_12_maxsteps_30661140_FINAL/epoch=1-step=265000.ckpt',  # Path to model checkpoint
    'output_path': 'data_with_embeddings.h5ad',  # Where to save the result, it is a new h5ad
    'output_dir': '.',  # Directory for any intermediate outputs
    'batch_size': 32,
    'max_seq_len': 1500, 
    'aux_tokens': 30, 
    'chunk_size': 1000, # to prevent OOM
    'num_workers': 4,
    'precision': 32,
    'embedding_layer': -1,  # Which layer to extract embeddings from (-1 for last layer)
    'embedding_name': 'embeddings'  # Name suffix for the embedding key in adata.obsm
}


# ## Load Data and Create Dataset

# In[3]:


model = ad.read_h5ad('/lustre/groups/ml01/projects/2023_nicheformer/data/data_to_tokenize/model.h5ad')


# In[4]:


# Set random seed for reproducibility
pl.seed_everything(42)

# Load data
adata = ad.read_h5ad(config['data_path'])
technology_mean = np.load(config['technology_mean_path'])

# format data properly with the model
adata = ad.concat([model, adata], join='outer', axis=0)
# dropping the first observation 
adata = adata[1:].copy()


# As a reference, the metadata tokens are 
# 
# modality_dict = {
#     'dissociated': 3,
#     'spatial': 4,}
# 
# specie_dict = {
#     'human': 5,
#     'Homo sapiens': 5,
#     'Mus musculus': 6,
#     'mouse': 6,}
# 
# technology_dict = {
#     "merfish": 7,
#     "MERFISH": 7,
#     "cosmx": 8,
#     "NanoString digital spatial profiling": 8,
#     "visium": 9,
#     "10x 5' v2": 10,
#     "10x 3' v3": 11,
#     "10x 3' v2": 12,
#     "10x 5' v1": 13,
#     "10x 3' v1": 14,
#     "10x 3' transcription profiling": 15, 
#     "10x transcription profiling": 15,
#     "10x 5' transcription profiling": 16,
#     "CITE-seq": 17, 
#     "Smart-seq v4": 18,
# }

# In[5]:


# Change accordingly

adata.obs['modality'] = 4 # Spatial, Does it mean that it uses spatial context or does it mean That I need to add a spatial context?
adata.obs['specie'] = 5 # human
adata.obs['assay'] = 7 #Merfish


# In[6]:


if 'nicheformer_split' not in adata.obs.columns:
    adata.obs['nicheformer_split'] = 'train'


adata = adata[:1000,:]


# Create dataset
dataset = NicheformerDataset(
    adata=adata,
    technology_mean=technology_mean,
    split='train',
    max_seq_len=1500,
    aux_tokens=config.get('aux_tokens', 30),
    chunk_size=config.get('chunk_size', 1000),
    metadata_fields={'obs': ['modality', 'specie', 'assay']}
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config.get('num_workers', 4),
    pin_memory=True
)


# ## Load Model and Set Up Trainer

# In[13]:


# Load pre-trained model
model = Nicheformer.load_from_checkpoint(checkpoint_path=config['checkpoint_path'], strict=False)
model.eval()  # Set to evaluation mode

# Configure trainer
trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    default_root_dir=config['output_dir'],
    precision=config.get('precision', 32),
)


# ## Extract Embeddings

# In[14]:


print("Extracting embeddings...")
embeddings = []
device = model.embeddings.weight.device

with torch.no_grad():
    for batch in tqdm(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Get embeddings from the model
        emb = model.get_embeddings(
            batch=batch,
            layer=config.get('embedding_layer', -1)  # Default to last layer
        )
        embeddings.append(emb.cpu().numpy())


# Concatenate all embeddings
embeddings = np.concatenate(embeddings, axis=0)


# ## Save Results

# In[ ]:


# Store embeddings in AnnData object
embedding_key = f"X_niche_{config.get('embedding_name', 'embeddings')}"
adata.obsm[embedding_key] = embeddings

# Save updated AnnData
adata.write_h5ad(config['output_path'])

print(f"Embeddings saved to {config['output_path']} in obsm['{embedding_key}']")

