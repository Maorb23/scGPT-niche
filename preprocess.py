import numpy as np
import argparse
import os
from pathlib import Path
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Preprocess:
    def __init__(self, dataset_path, describe=False):
        self.dataset_path = dataset_path
        self.describe = describe

    def normalize_data(self, data):
        """Normalize the data using log normalization."""
        data = np.log1p(data)
        return data

    def preprocess_data(self):
        """Preprocess the data by embedding it using scGPT."""
        logger.info("Loading dataset...")
        disease_adata = sc.read_h5ad(self.dataset_path)

        # Subset for colon tissue
        colon_index = disease_adata.obs.reset_index().query("tissue=='colon'").index.to_list()
        colon_adata = disease_adata[colon_index, :]

        logger.warning(f"Shape of colon dataset:, {colon_adata.shape}")
        logger.warning(f"Obs Columns in colon dataset: {colon_adata.obs.columns.tolist()}")
        logger.warning(f"Var Columns in colon dataset: {colon_adata.var.columns.tolist()}")
        logger.warning(f"X shape in colon dataset: {colon_adata.X.shape}")
        logger.warning(f" Cell Types in the data: \n {colon_adata.obs.cell_type.value_counts()}")
        logger.warning(f"Value counts of layers: \n {colon_adata.obs.Layer.value_counts()}")
        logger.warning(f"Age group: \n {colon_adata.obs['age group'].value_counts()}")
        logger.warning(f"Sex: \n {colon_adata.obs['sex'].value_counts()}")

        logger.info("Full describe...")
        full_describe = colon_adata.obs.describe(include = 'all')
        logger.info("N_genes and N_count the dataset...")
        N_genes = colon_adata.obs.describe()


        # Convert to dense
        colon_adata = anndata.AnnData(
            X=colon_adata.to_df().values,
            obs=colon_adata.obs.copy(),
            var=colon_adata.var.copy()
        )

        logger.info(f"Shape of colon dataset after conversion to dense: {colon_adata.shape}")
        #Save the processed data after creating data/processed folder
        if not os.path.exists("ydata/data/processed"):
            os.makedirs("ydata/data/processed")
        colon_adata.write_h5ad("ydata/data/processed/colon_adata.h5ad")
        return  colon_adata, full_describe, N_genes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data using scGPT.")
    parser.add_argument("--dataset_path", default="ydata/data/base_dataset.h5ad", help="Path to the dataset.")
    parser.add_argument("--describe", action="store_true", help="Whether to describe the dataset.")
    args = parser.parse_args()
    preprocessor = Preprocess(args.dataset_path,args.describe)
    preprocessor.preprocess_data()