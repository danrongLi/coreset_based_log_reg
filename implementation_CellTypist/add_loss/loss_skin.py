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
import tables
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
from scipy.sparse import csr_matrix
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




#logger.info("heca")
#selected_ada = anndata.read("/storage/home/dvl5760/scratch/heca_200k.h5ad")
selected_ada = anndata.read("/storage/home/dvl5760/scratch/scp-atlas-export.h5ad")
logger.info("skin Data")

#logger.info(f"heca shape before sampling: {adata.shape}")
#adata = celltypist.samples.downsample_adata(adata = adata, mode = "each",n_cells = 1000, by = "cell_type",random_state=42,return_index=False )
#selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = 200000, by = "celltype_coarse",random_state=42,return_index=False )

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

#selected_ada = anndata.read_h5ad("/storage/home/dvl5760/scratch/mouse_10000.h5ad")
#logger.info("done reading sub-sampled data")
#logger.info(f"after sub-sampling AnnData shape: {selected_ada.shape}")

#selected_ada = celltypist.samples.downsample_adata(adata = adata, mode = "total",n_cells = 5000, by = "cell_type",random_state=42,return_index=False )
#logger.info(f"After downsampling AnnData shape: {selected_ada.shape}")

#selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = 5000, by = "celltype",random_state=42,return_index=False )
#logger.info(f"After downsampling AnnData shape: {selected_ada.shape}")

#selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = 5000, by = "ClusterNm",random_state=42,return_index=False )
#logger.info(f"After downsampling AnnData shape: {selected_ada.shape}")

sc.pp.normalize_total(selected_ada, target_sum=1e4)
logger.info("done normalizing total counts")
#sc.pp.log1p(selected_ada)
selected_ada.X = sc.pp.log1p(selected_ada.X, copy=False)
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

adata_train.X = adata_train.X.astype('float32')
adata_test.X = adata_test.X.astype('float32')

batch_size=1000
logger.info("batch_size")
logger.info(batch_size)


start = time.time()
my_model = celltypist.train(adata_train, "celltype_coarse",alpha=1/(0.001*len(adata_train)), feature_selection=False, check_expression = True,use_SGD = True,mini_batch = True, batch_size=batch_size, batch_number=100)
end=time.time()
logger.info("seconds used for training")
logger.info(end-start)

#my_model = celltypist.train(adata_train, "celltype",n_jobs = -1, check_expression = True, feature_selection=False,use_SGD = True,mini_batch = True, max_iter = 100, batch_size=3, batch_number=10)

#start = time.time()
#my_model = celltypist.train(adata_train, "ClusterNm",n_jobs = -1, check_expression = True, feature_selection=False,use_SGD = True,mini_batch = True, max_iter = 100, batch_size=batch_size, batch_number=100)
#end = time.time()
#logger.info("seconds used:")
#logger.info(end-start)

logger.info("done creating the my_model")
#my_model.write(f'{models.models_path}/heca_adata_train.pkl')
#logger.info("done writting the model")
#logger.info(models.models_path)

#my_model = models.Model.load(model='heca_adata_train.pkl')
#logger.info("done loading the written model")




start = time.time()
predictions = celltypist.annotate(adata_test, model = my_model,majority_voting = False,mode = 'best match')
end = time.time()
taken = end-start
logger.info("done predictions with seconds")
logger.info(taken)

def custom_ovr_log_loss(true_labels, predicted_probs):
    """
    Custom one-vs-rest log loss calculation.

    Parameters:
    - true_labels: Array of true class indices.
    - predicted_probs: 2D array of predicted probabilities (shape: [n_samples, n_classes]).

    Returns:
    - Average one-vs-rest log loss.
    """
    n_samples, n_classes = predicted_probs.shape
    epsilon = 1e-15  # To avoid log(0)
    clipped_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    total_loss = 0.0

    # Loop over each class to calculate the one-vs-rest log loss
    for c in range(n_classes):
        y_binary = (true_labels == c).astype(int)
        class_loss = -np.sum(y_binary * np.log(clipped_probs[:, c]) + (1 - y_binary) * np.log(1 - clipped_probs[:, c]))
        total_loss += class_loss

    # Average over all samples and classes
    average_loss = total_loss / (n_samples)
    return average_loss

predicted_labels = predictions.predicted_labels.values  # Array of predicted labels from AnnotationResult
probability_matrix = predictions.probability_matrix.values  # Probability matrix
true_labels = adata_test.obs['celltype_coarse']  # Assuming true labels are stored here

class_names = predictions.probability_matrix.columns  # Get class names from probability matrix columns
true_labels_numeric = np.array([np.where(class_names == label)[0][0] for label in true_labels])

log_loss_value = custom_ovr_log_loss(true_labels_numeric, probability_matrix)
logger.info(f"Custom One-vs-Rest Log Loss: {log_loss_value}")

accuracy = accuracy_score(true_labels, predicted_labels)
logger.info(f"Accuracy: {accuracy}")

balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
logger.info(f"balabced acc on test data: {balanced_acc}")
macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
logger.info(f"macro_f1 score on test data: {macro_f1}")
report = classification_report(true_labels, predicted_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()

class_rows = report_df.index[~report_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]

report_df['accuracy_per_class'] = None
for label in class_rows:
    precision = report_df.at[label, 'precision']
    recall = report_df.at[label, 'recall']
    support = report_df.at[label, 'support']
    # True positives (TP)
    TP = recall * support
    # Per-class accuracy
    accuracy = TP / support  # This is effectively the recall
    report_df.at[label, 'accuracy_per_class'] = accuracy

file_name = f'/storage/home/dvl5760/work/our_log_reg/change_coreset/add_loss/test_train_split/reports/classification_report_skin_celltypist.csv'
report_df.to_csv(file_name, index=True)
logger.info("done saving the classification report")



logger.info("then lets find the log loss on train data")


predictions = celltypist.annotate(adata_train, model = my_model,majority_voting = False,mode = 'best match')
predicted_labels = predictions.predicted_labels.values  # Array of predicted labels from AnnotationResult
probability_matrix = predictions.probability_matrix.values  # Probability matrix
true_labels = adata_train.obs['celltype_coarse']  # Assuming true labels are stored here

class_names = predictions.probability_matrix.columns  # Get class names from probability matrix columns
true_labels_numeric = np.array([np.where(class_names == label)[0][0] for label in true_labels])

log_loss_value = custom_ovr_log_loss(true_labels_numeric, probability_matrix)
logger.info(f"Custom One-vs-Rest Log Loss for train: {log_loss_value}")

logger.info("all done")
