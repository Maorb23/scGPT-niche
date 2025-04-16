import numpy as np
import argparse
import os
from pathlib import Path
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from bokeh.io import show
import logging
from bokeh.plotting import figure, output_file, column
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, CustomJS
#import dropdown and customjs for interactivity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Plots:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.logger = logger

    def plot_most_expressed_genes(self, adata, n_genes=20):
    
        # Calculate mean expression for each gene across all cells
        if isinstance(adata.X, np.ndarray):
            mean_expr = adata.X.mean(axis=0)
        else:
            # For sparse matrices
            mean_expr = np.array(adata.X.mean(axis=0)).flatten()
        
        # Create a DataFrame with gene names and mean expression
        gene_expr_df = pd.DataFrame({
            'gene': adata.var.index,
            'mean_expression': mean_expr
        })
        
        # Sort by mean expression in descending order
        sorted_genes = gene_expr_df.sort_values('mean_expression', ascending=False)
        # Get top n genes
        top_genes = sorted_genes.head(n_genes)
        # Faster alternative using plain matplotlib
        fig = plt.figure(figsize=(12, 6))
        plt.bar(top_genes['gene'], top_genes['mean_expression'])
        plt.xticks(rotation=90)
        plt.title('Top 20 Most Expressed Genes')
        plt.tight_layout()
        return fig
    
    def plot_combined_distributions(self, adata, save_html=True):
        """Create interactive Bokeh plots with a dropdown to select."""

        from bokeh.plotting import figure, output_file
        from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Dropdown, Range1d
        from bokeh.layouts import column

        # Prepare data sources
        celltype_groups = adata.obs['Custom cell groups'].value_counts()
        celltypes_groups_source = ColumnDataSource(data={
            'x': celltype_groups.index.tolist(),
            'y': celltype_groups.values.tolist()
        })

        amit_clusters = adata.obs['clusters-amit'].value_counts()
        amit_clusters = amit_clusters[amit_clusters > 300]
        amit_clusters_source = ColumnDataSource(data={
            'x': amit_clusters.index.tolist(),
            'y': amit_clusters.values.tolist()
        })

        tumor_proximal = adata.obs['tumor_proximal'].fillna("NA").astype(str).value_counts()
        tumor_proximal_source = ColumnDataSource(data={
            'x': tumor_proximal.index.tolist(),
            'y': tumor_proximal.values.tolist()
        })

        region = adata.obs['region'].value_counts()
        region_source = ColumnDataSource(data={
            'x': region.index.tolist(),
            'y': region.values.tolist()
        })

        # Initialize with celltype_groups
        initial_source = celltypes_groups_source
        initial_title = "Cell Count per Cell Type"
        initial_xlabel = "Cell Type"

        p = figure(
            x_range=initial_source.data['x'],
            y_range=Range1d(start=0, end=max(initial_source.data['y']) * 1.1),
            title=initial_title,
            height=400, width=800, toolbar_location=None, tools="hover"
        )

        bars = p.vbar(x='x', top='y', source=initial_source, width=0.8)
        p.xaxis.axis_label = initial_xlabel
        p.yaxis.axis_label = "Cell Count"
        p.xaxis.major_label_orientation = 1.2
        p.xaxis.major_label_text_font_size = "8pt"
        p.yaxis.major_label_text_font_size = "10pt"

        hover = p.select_one(HoverTool)
        hover.tooltips = [("Label", "@x"), ("Count", "@y")]

        # Updated dropdown menu
        menu = [
            ("Cell Type", "celltype"),
            ("Amit Cluster", "amitcluster"),
            ("Tumor Proximity", "tumorprox"),
            ("Region", "region")
        ]

        dropdown = Dropdown(label="Select Plot", button_type="warning", menu=menu)

        # Updated callback to match actual variable meaning
        callback = CustomJS(args=dict(
            bars=bars,
            plot=p,
            source_celltype=celltypes_groups_source,
            source_amitcluster=amit_clusters_source,
            source_tumorprox=tumor_proximal_source,
            source_region=region_source
        ), code="""
            const selected = cb_obj.item;
            let new_source;
            if (selected === "celltype") {
                new_source = source_celltype;
                plot.title.text = "Cell Count per Cell Type";
            } else if (selected === "amitcluster") {
                new_source = source_amitcluster;
                plot.title.text = "Cell Count per Donor";
            } else if (selected === "tumorprox") {
                new_source = source_tumorprox;
                plot.title.text = "Cell Count per Tumor Proximity";
            } else if (selected === "region") {
                new_source = source_region;
                plot.title.text = "Cell Count per Region";
            }
            bars.data_source.data = new_source.data;
            plot.x_range.factors = new_source.data['x'];
            bars.data_source.change.emit();
            plot.change.emit();
        """)

        dropdown.js_on_event("menu_item_click", callback)

        layout = column(dropdown, p)

        if save_html:
            output_file("interactive_distribution.html")

        return layout



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script for scGPT-niche-clean")
    parser.add_argument("--dataset_path", type=str, default="ydata/data/Vizgen-hCRC-1313910_VS39.h5ad", help="Path to the dataset.")
    args = parser.parse_args()
    
    plot_class = Plots(args.dataset_path)
    colon_adata = sc.read(args.dataset_path)
    genes_plot = plot_class.plot_most_expressed_genes(colon_adata, 20)
    genes_plot.show()

    # Create interactive plot with dropdown
    interactive_plot = plot_class.plot_combined_distributions(colon_adata, save_html=True)
    show(interactive_plot)

    
    