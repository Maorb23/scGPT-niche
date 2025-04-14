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
        TLS_adata = sc.read_h5ad(self.dataset_path)
        logger.info("Dataset loaded successfully.")

        # Subset for colon tissue

        logger.warning(f"Shape of TLS dataset:, {TLS_adata.shape}")
        logger.warning(f"Obs Columns in TLS dataset: {TLS_adata.obs.columns.tolist()}")
        logger.warning(f"Var Columns in TLS dataset: {TLS_adata.var.columns.tolist()}")
        logger.warning(f"X shape in TLS dataset: {TLS_adata.X.shape}")
        logger.warning(f"Cell Groups in the data: \n {TLS_adata.obs['Custom cell groups'].value_counts()}")
        logger.warning(f"Cell Clusters in the data: \n {TLS_adata.obs['clusters-amit+DC'].value_counts()}")
        logger.warning(f"Descriptive statisics of X: \n {TLS_adata.obs['center_x'].describe()}")
        logger.warning(f"Descriptive statisics of Y: \n {TLS_adata.obs['center_y'].describe()}")
        logger.warning(f"Labels in the data: \n {TLS_adata.obs['region'].value_counts()/TLS_adata.shape[0]}")
        logger.warning(f"Tumor proximal: \n, {TLS_adata.obs['tumor_proximal'].value_counts()}")
        logger.warning(f"Immune-Epithel-stroma: \n {TLS_adata.obs['ImmuneEpithelStroma'].value_counts()}")
        logger.warning(f"Detailed Cell type: \n {TLS_adata.obs['detailed_cell_type'].value_counts()}")


        # Convert to dense
        TLS_adata = anndata.AnnData(
            X=TLS_adata.to_df().values,
            obs=TLS_adata.obs.copy(),
            var=TLS_adata.var.copy()
        )

        logger.info(f"Shape of colon dataset after conversion to dense: {TLS_adata.shape}")
        #Save the processed data after creating data/processed folder
        if not os.path.exists("ydata/data/processed"):
            os.makedirs("ydata/data/processed")
        # Save to processed directory with the same filename
        save_path = Path("ydata/data/processed") / Path(self.dataset_path).stem
        save_path = str(save_path) + "_colon.h5ad"
        TLS_adata.write_h5ad(save_path)
        logger.warning(f"Processed data saved to {save_path}")

        return  TLS_adata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data using scGPT.")
    parser.add_argument("--dataset_path", default="ydata/data/Vizgen-hCRC-1313910_VS39.h5ad", help="Path to the dataset.")
    parser.add_argument("--describe", action="store_true", help="Whether to describe the dataset.")
    args = parser.parse_args()
    preprocessor = Preprocess(args.dataset_path,args.describe)
    preprocessor.preprocess_data()
