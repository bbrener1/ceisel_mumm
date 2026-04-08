#%%
import scanpy as sc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from pathlib import Path

#%%
old_neighbors_path = "/home/bmb/haxx/working/ceisel_mumm/data/joint_annotated_old_neighbors.h5ad"
new_neighbors_path = "/home/bmb/haxx/working/ceisel_mumm/data/joint_annotated_new_neighbors.h5ad"

old_neighbors = sc.read_h5ad(old_neighbors_path)
new_neighbors = sc.read_h5ad(new_neighbors_path)

#%%
new_neighbors.uns['neighbors']
# %%
old_neighbors.uns['neighbors']
# %%
new_neighbors.obsp['connectivities']
# %%
for i,old,new in zip(np.arange(old_neighbors.shape[0]),old_neighbors.obsp['connectivities'],new_neighbors.obsp['connectivities']):
    old = old.toarray()
    new = new.toarray()
    if not np.allclose(old,new):
        print(f"Mismatch found:{i}")
        break
# %%
old = old_neighbors.obsp['connectivities'][0].toarray()[0]
new = new_neighbors.obsp['connectivities'][0].toarray()[0]

# %%
print(np.sum(old))
print(np.sum(new))
# %%
np.arange(old.shape[0])[old > 0]
# %%
np.arange(new.shape[0])[new > 0]
# %%
# We see that certain distances are weighted differently between the two
# Distance calculation relies on UMAP. Version mismatch, or are we off bc UMAP is an 
# explicit install in mid-torch?