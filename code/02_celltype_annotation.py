import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

script_loc = Path('01_load_and_preprocess.py').resolve().parent
data_loc = script_loc.parent / "data"
resources_loc = script_loc.parent / 'resources'


#Read in preprocessed data
cdata = sc.read_h5ad(data_loc / 'intermediate/c_preprocessed.h5ad')
pdata = sc.read_h5ad(data_loc / 'intermediate/p_preprocessed.h5ad')

#Format for mapmycells
#By dropping all data except for the central sparse matrix and the var and obs assignments
map_cdata = cdata.copy()
map_cdata.obs = map_cdata.obs[[]]
map_cdata.var = map_cdata.var[[]]
del map_cdata.uns
map_cdata.layers.clear(keep_x=True)


map_pdata = pdata.copy()
map_pdata.obs = map_pdata.obs[[]]
map_pdata.var = map_pdata.var[[]]
del map_pdata.uns
map_pdata.layers.clear(keep_x=True)


#Write mapping formated objects to data/intermediate
map_pdata.write_h5ad(data_loc / 'intermediate/map_pdata.h5ad')

map_cdata.write_h5ad(data_loc / 'intermediate/map_cdata.h5ad')


#Run MapMyCells through https://knowledge.brain-map.org/mapmycells/process 
#Using Hierarchical Mapping to 10x Whole Human Brain reference taxonomy for both files


#Read in pdata cell assignments, skipping metadata rows and removing machine readable cluster assignments
pdata_cells = (pd.read_csv(resources_loc / 'map_pdata_10xWholeHumanBrain(CCN202210140)_HierarchicalMapping_UTC_1793473199081.csv', 
                           skiprows=4))
pdata_cells = (pdata_cells.drop(columns=['supercluster_label', 'cluster_label', 'subcluster_label', 'subcluster_alias'])
                            .set_index('cell_id'))

#Convert categorical data to be categorical
pdata_cells = pdata_cells.astype(
    {'supercluster_name': 'category', 'cluster_name': 'category', 'subcluster_name': 'category'}
    )


#Read in cdata cell assignments, skipping metadata rows and removing machine readable cluster assignments
cdata_cells = (pd.read_csv(resources_loc / 'map_cdata_10xWholeHumanBrain(CCN202210140)_HierarchicalMapping_UTC_1793474000789.csv', 
                           skiprows=4))
cdata_cells = (cdata_cells.drop(columns=['supercluster_label', 'cluster_label', 'subcluster_label', 'subcluster_alias'])
                            .set_index('cell_id'))

#Convert categorical data to be categorical
cdata_cells = cdata_cells.astype(
    {'supercluster_name': 'category', 'cluster_name': 'category', 'subcluster_name': 'category'}
    )


#Add cell type assignments to pdata.obs and cdata.obs
pdata.obs = pdata.obs.merge(pdata_cells, left_index=True, right_index=True, how='left')

cdata.obs = cdata.obs.merge(cdata_cells, left_index=True, right_index=True, how='left')


#Find the top 2000 variable genes
sc.pp.highly_variable_genes(pdata, n_top_genes=2000, layer='CP10k_counts', batch_key='id')

sc.pp.highly_variable_genes(cdata, n_top_genes=2000, layer='CP10k_counts', batch_key='id')



#Find PCs
sc.tl.pca(pdata, mask_var="highly_variable", layer='CP10k_counts')

sc.tl.pca(cdata, mask_var="highly_variable", layer='CP10k_counts')

#Write pre-integration data to data/intermediate
pdata.write_h5ad(data_loc / 'intermediate/p_preintegration.h5ad')

cdata.write_h5ad(data_loc / 'intermediate/c_preintegration.h5ad')