import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

#Read in input data
script_loc = Path(__file__).resolve().parent
data_loc = script_loc.parent / "data"
resources_loc = script_loc.parent / 'resources'

adata = sc.read_h5ad(data_loc / "input/GSE225158_BU_OUD_Striatum_refined_all_SeuratObj_N22.h5ad")

#Drop irrelevant columns from adata.obs
adata.obs = (adata.obs.drop(columns=['nFeature_RNA', 'nCount_SCT', 'nFeature_SCT', 
                                     'integrated_snn_res.0.5', 
                                     'integrated_snn_res.2', 'integrated_snn_res.0.1', 
                                     'integrated_snn_res.1', 'seurat_clusters', 'level2',
                                     'nCount_RNA', 'percent.mt', 'n_genes']))

#Replace all periods with underscores, make all column names lowercase
adata.obs.columns = (adata.obs.columns
                     .str.replace('.', '_', regex=False)
                     .str.lower())


#Drop irrelevant columns from adata.var
adata.var = (adata.var.drop(columns=['highly_variable', 'dispersions',
                                     'features', 'dispersions_norm',
                                     'n_cells', 'means']))


#Drop all uns data and umap coordinates
del adata.uns
del adata.obsm["X_umap"]

#Annotate mitochondrial genes
adata.var['mt'] = adata.var.index.str.startswith('MT-')


#Bring in ribosomal reads list and annotate
ribo = pd.read_csv( resources_loc/ 'KEGG_RIBOSOME.v2026.2.Hs.txt', header=None)
adata.var['ribo'] = adata.var_names.isin(ribo[0].values)


#Copy raw data into a regular anndata object
adata_raw = adata.raw.to_adata()
adata_raw.var_names = adata_raw.var['features'].values


#Keep only the gene names that actually exist in adata_raw (to avoid dimension errors)
valid_vars = [name for name in adata.var_names if name in adata_raw.var_names]

#Slice the raw matrix data to the shared gene names only
aligned_raw_X = adata_raw[:, valid_vars].X

#Making raw count a dedicated layer
adata.layers['raw_counts'] = aligned_raw_X


#Calculate standard qc metrics on mt and ribo
#This includes total counts, mean counts, etc.
sc.pp.calculate_qc_metrics(adata, 
                            layer="raw_counts", 
                            qc_vars=['mt', 'ribo'], 
                            percent_top=None, 
                            log1p=False, 
                            inplace=True
                            )

#Downcast the underlying data arrays in-place to save memory
adata.layers["raw_counts"].data = adata.layers["raw_counts"].data.astype("float32")
adata.X.data = adata.X.data.astype("float32")

#Force Python to clean up any leftover float64 memory fragments
import gc
gc.collect()

#Seperate by region
cdata = adata[adata.obs['region'] == 'Caudate']
pdata = adata[adata.obs['region'] == 'Putamen']

#Insert raw_counts layer into the base
cdata.X = cdata.layers["raw_counts"]

#Normalize to CP10k, create a new layer
cdata.layers["CP10k_counts"] = cdata.X
sc.pp.normalize_total(cdata, target_sum=1e4, layer='CP10k_counts')
sc.pp.log1p(cdata, layer='CP10k_counts')

#Insert raw_counts layer into the base
pdata.X = pdata.layers["raw_counts"]

#Normalize to CP10k, create a new layer
pdata.layers["CP10k_counts"] = pdata.X
sc.pp.normalize_total(pdata, target_sum=1e4, layer='CP10k_counts')
sc.pp.log1p(pdata, layer='CP10k_counts')

#Delete raw slot
del cdata.raw
del pdata.raw

#Filter pdata by mt pct
pdata = pdata[pdata.obs['pct_counts_mt'] < 10]

#Write caudate and putamen objects to data/intermediate
cdata.write_h5ad(data_loc / 'intermediate/c_preprocessed.h5ad')
pdata.write_h5ad(data_loc / 'intermediate/p_preprocessed.h5ad')