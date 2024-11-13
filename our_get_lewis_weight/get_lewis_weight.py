import logging
from scipy import sparse
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

from sklearn.preprocessing import LabelEncoder
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


import gurobipy as gp
from gurobipy import GRB

#import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import inv


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


#logger.info("reading heca_200k")
adata = anndata.read("/storage/home/dvl5760/scratch/heca_200k.h5ad")

#adata = anndata.read_h5ad("/scratch/dvl5760/simonson_ready_for_jupyter_uniformed.h5ad")


#selected_ada = anndata.read("/storage/home/dvl5760/scratch/Zheng68K.h5ad")
#logger.info(f"Zhengdata AnnData shape: {selected_ada.shape}")


#selected_ada = anndata.read_h5ad("/storage/home/dvl5760/scratch/Macosko_Mouse_Atlas_Single_Nuclei.Use_Backed.h5ad", backed="r")
#logger.info(f"AnnData shape: {selected_ada.shape}")


adata = celltypist.samples.downsample_adata(adata = adata, mode = "total",n_cells = 10000, by = "cell_type",random_state=42,return_index=False )
#adata = celltypist.samples.downsample_adata(adata = adata, mode = "total",n_cells = 3000, by = "cell_type",random_state=42,return_index=False )
#adata = celltypist.samples.downsample_adata(adata = adata, mode = "total",n_cells = 3000, by = "cell_type",random_state=42,return_index=False )

logger.info("lets normalize and log1p this")
sc.pp.normalize_total(adata, target_sum=1e4)
logger.info("done normalizing total counts")
sc.pp.log1p(adata)
logger.info("done log1p transform")





def compute_lewis_weights(X):
    """
    Computes the Lewis weights for a sparse matrix X efficiently.
    
    Parameters:
    X (scipy.sparse.csr_matrix): A sparse matrix of shape (m, n).
    
    Returns:
    np.ndarray: An array of Lewis weights of size m.
    """
    # Ensure the input is a csr_matrix for efficient row operations
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)
        
    # Get the number of rows
    m, _ = X.shape
    
    # Initialize weights to 1
    weights = np.ones(m)
    
    # Convert the weights to a diagonal sparse matrix for efficiency
    W_inv_diag = diags(1 / weights)
    
    # Iteratively solve for the weights using fixed-point iterations
    max_iter = 100
    tol = 1e-6
    
    for _ in range(max_iter):
        # Compute the inverse of (X.T @ W^-1 @ X)
        M_inv = inv(X.T @ W_inv_diag @ X)
        
        # Calculate the new weights based on the definition
        new_weights = np.array([
            1 / np.sqrt(X[i, :].dot(M_inv.dot(X[i, :].T)).toarray().flatten()[0])
            for i in range(m)
        ])
        
        # Check for convergence
        if np.linalg.norm(new_weights - weights) < tol:
            break
        
        # Update weights
        weights = new_weights
        W_inv_diag = diags(1 / weights)
    
    return weights


X = adata.X


logger.info("start PCA")
pca = PCA(n_components=10, svd_solver='arpack')
pca.fit(X)
data_pca = pca.transform(X)
X = data_pca

logger.info("done PCA")

logger.info("begin calculating lewis weight")
weights = compute_lewis_weights(X) 
logger.info("done calculating")
df = pd.DataFrame(weights, columns=["Lewis_Weights"])
df.to_csv("/storage/home/dvl5760/work/lewis_weight/weights_heca_10k_10.csv", index=False)
logger.info("done saving csv")
