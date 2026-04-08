#%%
import scanpy as sc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from pathlib import Path

#%%
ann_path = "/home/bmb/haxx/working/ceisel_mumm/data/ceisel_adata_metadata.csv"
annotations = pd.read_csv(ann_path)
# %%
joint = sc.read_h5ad("/home/bmb/haxx/working/ceisel_mumm/data/joint_annotated.h5ad")
# %%
len(set(joint.obs_names))
# %%
some_cells = np.random.randint(0,len(annotations),size=20)
# %%
ann_sub = annotations.iloc[some_cells]
#%%
ann_sub
# %%
for i,cell,marked_condition in ann_sub[['Cell.ID','Condition']].itertuples():
    f1,f2,barcode = cell.split("_")
    print(f"{f1}_{f2}_{barcode}")
    print(marked_condition)
    print(joint[barcode].obs[['exp_condition','time']])
# %%
a,b = list(ann_sub[['Cell.ID','Condition']].itertuples())[0]
# %%
