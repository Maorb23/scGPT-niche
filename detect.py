from pathlib import Path
import json
import scanpy as sc

# Path to the model directory (adjust as needed)
model_dir = Path("ydata/models/best_model")

# Read the vocabulary file (assuming it is a JSON list of gene names)
vocab_file = model_dir / "vocab.json"  # Adjust filename if necessary
with open(vocab_file, "r") as f:
    model_vocab = set(json.load(f))

# Path to your AnnData file
adata_path = "ydata/data/processed/Vizgen-hCRC-1313910_VS39_colon.h5ad"  # Adjust the path as needed
colon_adata = sc.read_h5ad(adata_path)

# Get the list of genes from your AnnData object
all_genes = set(colon_adata.var.index)

# Determine which genes are NOT in the model vocabulary
unrecognized_genes = all_genes - model_vocab
print("Genes not recognized by scGPT:", unrecognized_genes)

