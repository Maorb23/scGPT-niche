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

    def plot_sex_distribution(self, colon_adata):
        """Plot and log cell distribution by sex."""
        sex_counts = colon_adata.obs['sex'].value_counts()
        fig = plt.figure(figsize=(4, 4))
        sex_counts.plot(kind='bar')
        plt.title('Cell Distribution by Sex')
        plt.xlabel('Sex')
        plt.ylabel('Count')
        return fig

    def plot_most_expressed_genes(self, adata, n_genes=20):
    
        # Calculate mean expression for each gene across all cells
        if isinstance(adata.X, np.ndarray):
            mean_expr = adata.X.mean(axis=0)
        else:
            # For sparse matrices
            mean_expr = np.array(adata.X.mean(axis=0)).flatten()
        
        # Create a DataFrame with gene names and mean expression
        gene_expr_df = pd.DataFrame({
            'gene': adata.var['feature_name'],
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

    def number_cells_per_donor(self, colon_adata):
        """Plot the number of cells per donor."""
        donor_counts = colon_adata.obs['donor_id'].value_counts()
        fig = plt.figure(figsize=(8, 4))
        donor_counts.plot(kind='bar')
        plt.title("Number of Cells per Donor")
        plt.xlabel("Donor ID")
        plt.ylabel("Cell Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_cell_type_distribution(self, colon_adata):
        """Plot the distribution of cell types."""
        celltype_counts = colon_adata.obs['Celltype'].value_counts()
        celltype_sorted = celltype_counts[celltype_counts > 300]
        fig = plt.figure(figsize=(10, 4))
        celltype_sorted.plot(kind='bar',fontsize=6.5)
        plt.title("Cell Count per Cell Type")
        plt.xlabel("Cell Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_combined_distributions(self, colon_adata, save_html=True):
        """Create interactive Bokeh plots with a dropdown to select."""
        
        # Prepare data sources
        celltype_counts = colon_adata.obs['Celltype'].value_counts()
        celltype_sorted = celltype_counts[celltype_counts > 300]
        celltype_source = ColumnDataSource(data={
            'x': celltype_sorted.index.tolist(),
            'y': celltype_sorted.values.tolist()
        })

        donor_counts = colon_adata.obs['donor_id'].value_counts()
        donor_source = ColumnDataSource(data={
            'x': donor_counts.index.tolist(),
            'y': donor_counts.values.tolist()
        })

        sex_counts = colon_adata.obs['sex'].value_counts()
        sex_source = ColumnDataSource(data={
            'x': sex_counts.index.tolist(),
            'y': sex_counts.values.tolist()
        })

        layer_counts = colon_adata.obs['Layer'].value_counts()
        layer_source = ColumnDataSource(data={
            'x': layer_counts.index.tolist(),
            'y': layer_counts.values.tolist()
        })

        # Create the base figure (empty)
        p = figure(x_range=celltype_sorted.index.tolist(),
                   title="Cell Count per Cell Type",
                   height=400, width=800, toolbar_location=None, tools="hover")

        bars = p.vbar(x='x', top='y', source=celltype_source, width=0.8)
        p.xaxis.major_label_orientation = 1.2
        p.xaxis.major_label_text_font_size = "8pt"
        p.yaxis.major_label_text_font_size = "10pt"

        hover = p.select_one(HoverTool)
        hover.tooltips = [("Label", "@x"), ("Count", "@y")]

        # Create a Dropdown menu
        menu = [("Cell Type", "celltype"),
                ("Donor", "donor"),
                ("Sex", "sex"),
                ("Layer", "layer")]
        dropdown = Dropdown(label="Select Plot", button_type="warning", menu=menu)

        # Link dropdown selection to change data
        callback = CustomJS(args=dict(source=celltype_source, 
                                      source_donor=donor_source, 
                                      source_sex=sex_source, 
                                      source_layer=layer_source,
                                      bars=bars, plot=p),
                            code="""
            const selected = cb_obj.item;
            let new_source;
            if (selected === "celltype") {
                new_source = source;
                plot.title.text = "Cell Count per Cell Type";
            } else if (selected === "donor") {
                new_source = source_donor;
                plot.title.text = "Cell Count per Donor";
            } else if (selected === "sex") {
                new_source = source_sex;
                plot.title.text = "Cell Distribution by Sex";
            } else if (selected === "layer") {
                new_source = source_layer;
                plot.title.text = "Cell Distribution by Layer";
            }
            bars.data_source.data = new_source.data;
            plot.x_range.factors = new_source.data['x'];
            bars.data_source.change.emit();
            plot.change.emit();
        """)

        dropdown.js_on_event("menu_item_click", callback)

        layout = column(dropdown, p)

        # Save to HTML
        if save_html:
            output_file("interactive_distribution.html")

        return layout
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting script for scGPT-niche-clean")
    parser.add_argument("--dataset_path", type=str, default="ydata/data/base_dataset.h5ad", help="Path to the dataset.")
    args = parser.parse_args()
    
    plot_class = Plots(args.dataset_path)
    colon_adata = sc.read(args.dataset_path)
    genes_plot = plot_class.plot_most_expressed_genes(colon_adata, 20)
    genes_plot.show()

    # Create interactive plot with dropdown
    interactive_plot = plot_class.plot_combined_distributions(colon_adata, save_html=True)
    show(interactive_plot)

    
    