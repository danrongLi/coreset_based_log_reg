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

# Check system memory
#def check_memory():
#    mem = psutil.virtual_memory()
#    logger.info(f"Total memory: {mem.total / (1024 ** 3):.2f} GB")
#    logger.info(f"Available memory: {mem.available / (1024 ** 3):.2f} GB")
#    if mem.available < (10 * 1024 ** 3):  # less than 10 GB available
#        logger.warning("Available memory is low. Consider adding swap space or freeing up memory.")

#check_memory()

# Ensure the file is correctly read
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

#logger.info(len((np.unique(adata_entire.obs_names))))
#logger.info(len(adata_entire.obs["cell_type"].unique()))

#num_per_cell = int(200000/len(adata_entire.obs["cell_type"].unique()))
#logger.info("num_per_cell: "+str(num_per_cell))

adata = anndata.read("/Users/danrongli/Desktop/Feature_Space_Logistic/server_results/oct23/heca_200k.h5ad")

#logger.info("this time, we use celltypist to select our heca 200000 data!")

#logger.info("then lets downsample so that each label has 100 samples")

#adata = celltypist.samples.downsample_adata(adata = adata_entire, mode = "total",n_cells=200000, by = "cell_type",random_state=42,return_index=False, balance_cell_type=True)

#adata = celltypist.samples.downsample_adata(adata = adata, mode = "each",n_cells = 100, by = "cell_type",random_state=44,return_index=False )


logger.info(f"heca shape after sampling: {adata.shape}")

#logger.info("Done reading heca")

selected_ada= anndata.read_h5ad("/Users/danrongli/Desktop/Feature_Space_Logistic/server_results/oct23/simonson_ready_for_jupyter_uniformed.h5ad")

#logger.info("Done reading simonson")

#logger.info("Now lets transform the heca data to fit into celltypist")

logger.info("lets normalize and log1p this")
sc.pp.normalize_total(adata, target_sum=1e4)
logger.info("done normalizing total counts")
sc.pp.log1p(adata)
logger.info("done log1p transform")
logger.info("Done transforming for heca")


logger.info(f"Simonson AnnData shape: {selected_ada.shape}")



highly_variable_genes = adata.var_names
common_genes = list(set(highly_variable_genes).intersection(set(selected_ada.var_names)))
logger.info("common_genes")
logger.info(common_genes)
selected_ada = selected_ada[:, common_genes]

logger.info(f"After selecting genes Simonson AnnData shape: {selected_ada.shape}")


# Normalize the total counts per cell to 10,000
sc.pp.normalize_total(selected_ada, target_sum=1e4)
logger.info("done normalizing total counts")
# Log1p transform the data
sc.pp.log1p(selected_ada)
logger.info("done log1p transform")

#logger.info("Done transforming for simonson")

# logger.info("lets save the selected_ada and use this in scMulan")
# selected_ada.write("ready_for_input_scMulan.h5ad")
# logger.info("done saving the simonson data for scMulan")


logger.info("lets skip the feature selection")

#logger.info("lets do feature selection first")
#model_fs = celltypist.train(adata, "cell_type", n_jobs = 10, max_iter = 5, use_SGD = True)
#logger.info("done creating model_fs")
#gene_index = np.argpartition(np.abs(model_fs.classifier.coef_), -50, axis = 1)[:, -50:]
#logger.info("done finding gene_index")
#gene_index = np.unique(gene_index)
#logger.info("done making it unique")
#logger.info("number of genes selected: "+str(len(gene_index)))
#logger.info("original number of genes: "+str(len(adata.var_names)))

#genes_list = gene_index


#genes_list = adata.var_names
#logger.info("lets have a peek of the genes_list")
#logger.info(genes_list[:5])

#my_model = celltypist.train(adata[:,genes_list], "cell_type",n_jobs = 10, check_expression = False, max_iter = 100)

#my_model = celltypist.train(adata, "cell_type",n_jobs = -1, check_expression = True, feature_selection=False, max_iter = 100)

#batch_sizes = [5,10,50,100,500,1000]

#batch_sizes = [2000, 30, 3, 7, 20, 40]
#batch_sizes = [3,5,7,10,20,40,50,100,500,1000,2000,30]

batch_sizes = [1000]
batch_number = 100

for batch_size in batch_sizes:
    logger.info("batch_size")
    logger.info(batch_size)
    start = time.time()
    my_model = celltypist.train(adata, "cell_type",n_jobs = -1, check_expression = True, feature_selection=False,use_SGD = True,mini_batch = True, max_iter = 100, batch_size=batch_size, batch_number=batch_number)
    logger.info("done creating the my_model")
    end = time.time()
    logger.info("seconds used:")
    logger.info(end-start)
    #my_model.write(f'{models.models_path}/heca_pretrain_model_uniformed_correct_simonson.pkl')
    #logger.info("done writting the model")
    #logger.info(models.models_path)

    #my_model = models.Model.load(model='heca_pretrain_model_uniformed_correct_simonson.pkl')
    #logger.info("done loading the written model")


    predictions = celltypist.annotate(selected_ada, model = my_model,majority_voting = True,mode = 'best match')
    logger.info("done predictions")

    adata_result = predictions.to_adata(insert_prob = True)
    # logger.info("done getting predictions")
    # adata_result.write(f"/storage/home/dvl5760/work/celltypist/train_heca_test_simonson/majority_voting_batch_size_{batch_size}.h5ad")
    # logger.info("done write")
    
    #majority_voting
    celltypist_true = adata_result.obs["cell_type"].values.astype(str)
    celltypist_pred = adata_result.obs["predicted_labels"].values.astype(str)
    # celltypist_pred = adata_result.obs["majority_voting"].values.astype(str)
    celltypist_df = pd.DataFrame({'True_Labels': celltypist_true,'Predicted_Labels': celltypist_pred})

    before_acc = accuracy_score(celltypist_df["Predicted_Labels"].values, celltypist_df["True_Labels"].values)
    logger.info("before acc:")
    logger.info(before_acc)

    celltypist_df.loc[((celltypist_df["Predicted_Labels"] == "Basement membrane fibroblast") | (celltypist_df["Predicted_Labels"] == "Myofibroblast") | (celltypist_df["Predicted_Labels"] == "Alveolar fibroblast")| (celltypist_df["Predicted_Labels"] == "Adventitial fibroblast")) & (celltypist_df["True_Labels"] == "Fibroblast"), "Predicted_Labels"] = "Fibroblast"

    celltypist_df.loc[((celltypist_df["Predicted_Labels"] == "Capillary endothelial cell")| (celltypist_df["Predicted_Labels"] == "Lymphatic endothelial cell") | (celltypist_df["Predicted_Labels"] == "Vascular endothelial cell")|(celltypist_df["Predicted_Labels"] == "Sinusoidal endothelial cell")) & (celltypist_df["True_Labels"] == "Endothelial cell"), "Predicted_Labels"] = "Endothelial cell"

    celltypist_df.loc[((celltypist_df["Predicted_Labels"] == "Memory CD4 T cell") | (celltypist_df["Predicted_Labels"] == "T cell")|(celltypist_df["Predicted_Labels"] == "Naive CD8 T cell")) & (celltypist_df["True_Labels"] == "Lymphoid cell"), "Predicted_Labels"] = "Lymphoid cell"

    celltypist_df.loc[(celltypist_df["Predicted_Labels"] == "Vascular smooth muscle cell") & (celltypist_df["True_Labels"] == "Smooth muscle cell"), "Predicted_Labels"] = "Smooth muscle cell"


    after_acc = accuracy_score(celltypist_df["Predicted_Labels"].values, celltypist_df["True_Labels"].values)
    logger.info("after acc:")
    logger.info(after_acc)


    #celltypist.dotplot(predictions, use_as_reference = 'cell_type', use_as_prediction = 'majority_voting')
    #plt.savefig("/storage/home/dvl5760/work/new_simonson/results/pretrain_simonson_cell_type_dotplot_uniformed_best_small_100_after_200kheca.pdf", bbox_inches='tight')


#    sc.settings.figdir = "/storage/home/dvl5760/work/new_simonson/"
#    sc.pp.neighbors(selected_ada, n_neighbors=10, n_pcs=40)
#    sc.tl.umap(selected_ada)
#    logger.info("done calling umap")

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

