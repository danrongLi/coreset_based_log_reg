import logging
import os
import celltypist
from celltypist import models
import numpy as np
import pandas as pd
from scipy.linalg import norm
import scanpy as sc
from matplotlib.lines import Line2D
import random
# import tables
import h5py
import psutil  # For checking system memory
from sklearn.metrics import accuracy_score


from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import time

from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import diags
from scipy.linalg import svd, lstsq
from scipy.sparse.linalg import inv, LinearOperator, spsolve, factorized

import anndata
import itertools

#import scMulan
#from scMulan import GeneSymbolUniform


# Create or get the logger
logger = logging.getLogger(__name__)

# Set the level of the logger. This is optional and can be set to other levels (e.g., DEBUG, ERROR)
logger.setLevel(logging.INFO)

# Create a console handler and set the level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set the formatter for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

logger.info('done importing stuff')

#try:
#    file_path = '/storage/home/dvl5760/work/data/celldfhvg.h5'
#    h5 = pd.HDFStore(file_path, 'r')
#    logger.info("done h5")
#    data = h5['data']
#    logger.info("done data")
#    h5.close()
#except Exception as e:
#    logger.info(f"Error reading HDF5 file with pandas: {e}")
#    raise

# Read metadata and create AnnData object
#try:
#    metadata = pd.read_csv('/storage/home/dvl5760/work/data/meta_data.csv', index_col=0)
#    logger.info("done metadata")
#    adata_entire = anndata.AnnData(data)
#    logger.info("done adata")
#    adata_entire.obs = metadata
#    logger.info("done adata.obs")
#    logger.info("lets have a look at the adata")
#    logger.info(f"heca shape: {adata_entire.shape}")
#except Exception as e:
#    logger.info("An error occurred while reading the AnnData object:")
#    logger.info(e)
#    raise




#selected_ada = anndata.read("/storage/home/dvl5760/scratch/heca_200k.h5ad")
#logger.info(f"heca shape before sampling: {adata.shape}")
#adata = celltypist.samples.downsample_adata(adata = adata, mode = "each",n_cells = 1000, by = "cell_type",random_state=42,return_index=False )
#selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = 50000, by = "cell_type",random_state=42,return_index=False )

#logger.info(f"heca shape after sampling: {adata.shape}")

#logger.info("Done reading heca")

#selected_ada= anndata.read_h5ad("/scratch/dvl5760/simonson_ready_for_jupyter_uniformed.h5ad")

#logger.info("Done reading simonson")

#logger.info("Now lets transform the heca data to fit into celltypist")

#logger.info("lets normalize and log1p this")
#sc.pp.normalize_total(adata, target_sum=1e4)
#logger.info("done normalizing total counts")
#sc.pp.log1p(adata)
#logger.info("done log1p transform")
#logger.info("Done transforming for heca")


#logger.info(f"Simonson AnnData shape: {selected_ada.shape}")



#highly_variable_genes = adata.var_names
#common_genes = list(set(highly_variable_genes).intersection(set(selected_ada.var_names)))
#logger.info("common_genes")
#logger.info(common_genes)
#selected_ada = selected_ada[:, common_genes]

#logger.info(f"After selecting genes Simonson AnnData shape: {selected_ada.shape}")


#selected_ada = anndata.read("/storage/home/dvl5760/scratch/Zheng68K.h5ad")
#logger.info(f"Zhengdata AnnData shape: {selected_ada.shape}")


#selected_ada = anndata.read_h5ad("/storage/home/dvl5760/scratch/Macosko_Mouse_Atlas_Single_Nuclei.Use_Backed.h5ad", backed="r")
#logger.info(f"AnnData shape: {selected_ada.shape}")

#n_samples = 50000
#random_indices = np.random.choice(selected_ada.n_obs, size=n_samples, replace=False)
#logger.info("done random_indices")

#subset_ada = selected_ada[random_indices, :].to_memory()
#logger.info("done loading subset to memory")

#new_filename = "/storage/home/dvl5760/scratch/mouse_50000.h5ad"
#subset_ada.write(filename=new_filename)
#logger.info("done saving to scratch")

# selected_ada = anndata.read_h5ad("/storage/home/dvl5760/scratch/mouse_10000.h5ad")
selected_ada = anndata.read_h5ad("/Users/danrongli/Desktop/Feature_Space_Logistic/server_results/oct23/mouse_10000.h5ad")
#logger.info("done reading sub-sampled data")
#logger.info(f"after sub-sampling AnnData shape: {selected_ada.shape}")
logger.info("mouse")
#selected_ada = celltypist.samples.downsample_adata(adata = adata, mode = "total",n_cells = 5000, by = "cell_type",random_state=42,return_index=False )
#logger.info(f"After downsampling AnnData shape: {selected_ada.shape}")

#selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = 5000, by = "celltype",random_state=42,return_index=False )
#logger.info(f"After downsampling AnnData shape: {selected_ada.shape}")

#selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = 5000, by = "ClusterNm",random_state=42,return_index=False )
#logger.info(f"After downsampling AnnData shape: {selected_ada.shape}")

sc.pp.normalize_total(selected_ada, target_sum=1e4)
logger.info("done normalizing total counts")
sc.pp.log1p(selected_ada)
logger.info("done log1p transform")

#logger.info("Done transforming for simonson")

#logger.info("lets save the selected_ada and use this in scMulan")
#selected_ada.write("ready_for_scMulan_100_after_200kheca.h5ad")
#logger.info("done saving the simonson data for scMulan")



#my_model = celltypist.train(adata, "cell_type",n_jobs = -1, check_expression = True, feature_selection=False, max_iter = 100)


#logger.info("split heca data into 80% train and 20% test")
#train_indices, test_indices = train_test_split(range(adata.n_obs), test_size=0.2, random_state=42)
#adata_train = adata[train_indices].copy()
#adata_test = adata[test_indices].copy()


logger.info("split simonson data into 80% train and 20% test")
train_indices, test_indices = train_test_split(range(selected_ada.n_obs), test_size=0.2, random_state=42)
adata_train = selected_ada[train_indices].copy()
adata_test = selected_ada[test_indices].copy()


batch_size=1000
logger.info("batch_size")
logger.info(batch_size)

#my_model = celltypist.train(adata_train, "cell_type",n_jobs = -1, check_expression = True, feature_selection=False,use_SGD = True,mini_batch = True, max_iter = 100, batch_size=batch_size, batch_number=10)

#my_model = celltypist.train(adata_train, "celltype",n_jobs = -1, check_expression = True, feature_selection=False,use_SGD = True,mini_batch = True, max_iter = 100, batch_size=3, batch_number=10)

start = time.time()
my_model = celltypist.train(adata_train, "ClusterNm",n_jobs = -1, check_expression = True, feature_selection=False,use_SGD = True,mini_batch = True, max_iter = 100, batch_size=batch_size, batch_number=100)
end = time.time()
logger.info("seconds used:")
logger.info(end-start)

logger.info("done creating the my_model")
#my_model.write(f'{models.models_path}/heca_adata_train.pkl')
#logger.info("done writting the model")
#logger.info(models.models_path)

#my_model = models.Model.load(model='heca_adata_train.pkl')
#logger.info("done loading the written model")


predictions = celltypist.annotate(adata_test, model = my_model,majority_voting = False,mode = 'best match')
logger.info("done predictions")

adata_result = predictions.to_adata(insert_prob = True)

logger.info("done getting predictions")
# adata_result.write("/storage/home/dvl5760/work/celltypist/mouse_10k_21899_batchsize_1000_batch_num_100.h5ad")
logger.info("done write")


acc = accuracy_score(adata_result.obs["predicted_labels"].values, adata_result.obs["ClusterNm"].values)

logger.info("acc is")
logger.info(acc)

#celltypist.dotplot(predictions, use_as_reference = 'ClusterNm', use_as_prediction = 'predicted_labels')
#plt.savefig("/storage/home/dvl5760/work/celltypist/mouse_10k_21899_batchsize_1000_batch_num_10.pdf", bbox_inches='tight')


#sc.settings.figdir = "/storage/home/dvl5760/work/new_simonson/"
#sc.pp.neighbors(selected_ada, n_neighbors=10, n_pcs=40)
#sc.tl.umap(selected_ada)
#logger.info("done calling umap")

# Function to create legend handles
#def create_legend(ax, adata, color_key):
#    categories = adata.obs[color_key].cat.categories
#    colors = [adata.uns[color_key + '_colors'][i] for i in range(len(categories))]
#    legend_elements = [Line2D([0], [0], marker='o', color='w', label=cat,
#                              markerfacecolor=col, markersize=10) for cat, col in zip(categories, colors)]
#    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot cell_type
#fig, ax = plt.subplots(figsize=(15, 10))
#sc.pl.umap(selected_ada, color='cell_type', legend_loc=None, show=False, ax=ax)
#create_legend(ax, selected_ada, 'cell_type')
#plt.savefig("/storage/home/dvl5760/work/new_simonson/results/pretrain_simonson_cell_type_uniformed_best_small_100_after_200kheca.pdf", bbox_inches='tight')
#logger.info("done saving cell_type plot")

# Plot majority_voting
#fig, ax = plt.subplots(figsize=(15, 10))
#sc.pl.umap(selected_ada, color='majority_voting', legend_loc=None, show=False, ax=ax)
#create_legend(ax, selected_ada, 'majority_voting')
#plt.savefig("/storage/home/dvl5760/work/new_simonson/results/pretrain_simonson_majority_voting_uniformed_best_small_100_after_200kheca.pdf", bbox_inches='tight')
#logger.info("done saving majority_voting plot")

# Plot predicted_labels
#fig, ax = plt.subplots(figsize=(15, 10))
#sc.pl.umap(selected_ada, color='predicted_labels', legend_loc=None, show=False, ax=ax)
#create_legend(ax, selected_ada, 'predicted_labels')
#plt.savefig("/storage/home/dvl5760/work/new_simonson/results/pretrain_simonson_umap_predicted_uniformed_best_small_100_after_200kheca.pdf", bbox_inches='tight')
#logger.info("done saving predicted_labels plot")

logger.info("all done")

