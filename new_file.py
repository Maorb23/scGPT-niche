import scanpy as sc
adata = sc.read_h5ad("ydata/data/processed/base_dataset_colon.h5ad")
print(adata.shape)
print(adata.obs["tissue"].unique())
