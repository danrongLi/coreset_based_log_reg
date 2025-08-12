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


#logger.info("this time, we use celltypist to select our heca 200000 data!")

#logger.info("then lets downsample so that each label has 25 samples")

#adata = celltypist.samples.downsample_adata(adata = adata_entire, mode = "total",n_cells=200000, by = "cell_type",random_state=42,return_index=False, balance_cell_type=True)

#adata.write("/storage/home/dvl5760/scratch/heca_200k.h5ad")
#logger.info("done writing the heca_200k")


#logger.info("reading heca_200k")
adata = anndata.read("/storage/home/dvl5760/scratch/heca_200k.h5ad")

adata = celltypist.samples.downsample_adata(adata = adata, mode = "total",n_cells = 10000, by = "cell_type",random_state=42,return_index=False )


logger.info(f"heca shape after sampling: {adata.shape}")

#logger.info("Done reading heca")


logger.info("lets normalize and log1p this")
sc.pp.normalize_total(adata, target_sum=1e4)
logger.info("done normalizing total counts")
sc.pp.log1p(adata)
logger.info("done log1p transform")
logger.info("Done transforming for heca")

#logger.info("reading simonson, the things in between logger are all need to be comment back for reading heca")
#adata = anndata.read("/storage/home/dvl5760/work/new_simonson/ready_for_scMulan_100_after_200kheca.h5ad")
#adata = celltypist.samples.downsample_adata(adata = adata, mode = "each",n_cells = 25, by = "cell_type",random_state=42,return_index=False )
#adata = celltypist.samples.downsample_adata(adata = adata, mode = "total",n_cells=10000, by = "cell_type",random_state=42,return_index=False, balance_cell_type=False)

def create_model_previous(X_input,y_input, n_input):
    # Performing operations on matrices
    Dy = np.diag(y_input)
    DyX = Dy @ X_input
    Q, R = np.linalg.qr(DyX, "reduced")
    p = Q @ Q.T
    C = 1.0
    one_matrix = np.ones((n_input,n_input))
    one_diagonal = np.diag(np.diag(one_matrix))
    
    # Creating a Gurobi model
    this_model = gp.Model()
    
    # Creating variables
    this_z_plus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_plus_{this_i}", lb=1e-16) for this_i in range(n_input)]
    this_z_minus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_minus_{this_i}", lb=1e-16) for this_i in range(n_input)]
    
    # Adding the norm constraint
    norm_z = this_model.addVar()
    this_model.addConstr(norm_z == gp.quicksum(this_z_plus[this_i] + this_z_minus[this_i] for this_i in range(n_input)))
    this_model.addConstr(norm_z <= C)

    # Adding the element-wise constraints
    for this_i in range(n_input):
        constraint_expr = gp.LinExpr()
        for this_j in range(n_input):
            constraint_expr.addTerms(one_diagonal[this_i, this_j] - p[this_i, this_j], this_z_plus[this_j])
            constraint_expr.addTerms(-(one_diagonal[this_i, this_j] - p[this_i, this_j]), this_z_minus[this_j])
        this_model.addConstr(constraint_expr == 0, f"constraint_{this_i}")
    
    # Defining the objective function
    obj_expr = gp.LinExpr()
    obj_expr.add(gp.quicksum(this_z_plus[this_i] - this_z_minus[this_i] for this_i in range(n_input)))
    this_model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    return this_model, this_z_plus, this_z_minus, DyX, p


def create_model_previous2(X_input, y_input, n_input):
    # Step 1: Perform operations on the matrices
    Dy = np.diag(y_input)
    DyX = Dy @ X_input
    
    # QR decomposition (DyX may still be large, but unavoidable here)
    Q, R = np.linalg.qr(DyX, "reduced")
    
    # Projection matrix p, still dense but we will try to optimize its usage
    p = Q @ Q.T
    
    # Use a constant value for the diagonal matrix instead of creating full dense matrices
    C = 1.0
    one_diagonal = np.ones(n_input)  # Diagonal of identity matrix (1s)
    
    # Step 2: Create Gurobi model
    this_model = gp.Model()
    
    # Step 3: Create variables
    this_z_plus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_plus_{this_i}", lb=1e-16) for this_i in range(n_input)]
    this_z_minus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_minus_{this_i}", lb=1e-16) for this_i in range(n_input)]
    
    # Step 4: Add the norm constraint
    norm_z = this_model.addVar()
    this_model.addConstr(norm_z == gp.quicksum(this_z_plus[this_i] + this_z_minus[this_i] for this_i in range(n_input)))
    this_model.addConstr(norm_z <= C)

    # Step 5: Add element-wise constraints
    for this_i in range(n_input):
        constraint_expr = gp.LinExpr()
        for this_j in range(n_input):
            # No need to create one_matrix, just use the value directly (it's identity matrix)
            value = 1 if this_i == this_j else 0
            
            # Adding the terms based on one_diagonal
            constraint_expr.addTerms(value - p[this_i, this_j], this_z_plus[this_j])
            constraint_expr.addTerms(-(value - p[this_i, this_j]), this_z_minus[this_j])
        this_model.addConstr(constraint_expr == 0, f"constraint_{this_i}")

    # Step 6: Define the objective function
    obj_expr = gp.LinExpr()
    obj_expr.add(gp.quicksum(this_z_plus[this_i] - this_z_minus[this_i] for this_i in range(n_input)))
    this_model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    return this_model, this_z_plus, this_z_minus, DyX, p



def create_model_previous3(X_input, y_input, n_input):
    # Step 1: Perform operations on the matrices
    Dy = np.diag(y_input)
    DyX = Dy @ X_input
    
    # QR decomposition (DyX may still be large, but unavoidable here)
    Q, R = np.linalg.qr(DyX, "reduced")
    
    # Projection matrix p, converting to sparse if it has a lot of zeros
    p = Q @ Q.T
    p_sparse = sparse.csr_matrix(p)  # Convert to sparse matrix for efficiency
    
    # Use a constant value for the diagonal matrix instead of creating full dense matrices
    C = 1.0
    one_diagonal = np.ones(n_input)  # Diagonal of identity matrix (1s)

    # Step 2: Create Gurobi model
    this_model = gp.Model()

    # Set memory limit to avoid out-of-memory errors
    this_model.setParam('MemLimit', 100000)  # Limit to 100GB, adjust as necessary

    # Step 3: Create variables
    this_z_plus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_plus_{this_i}", lb=1e-16) for this_i in range(n_input)]
    this_z_minus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_minus_{this_i}", lb=1e-16) for this_i in range(n_input)]

    # Step 4: Add the norm constraint
    norm_z = this_model.addVar()
    this_model.addConstr(norm_z == gp.quicksum(this_z_plus[this_i] + this_z_minus[this_i] for this_i in range(n_input)))
    this_model.addConstr(norm_z <= C)

    # Step 5: Add element-wise constraints, with a tolerance to skip small coefficients
    tolerance = 1e-12  # Set a tolerance to filter out small values
    for this_i in range(n_input):
        constraint_expr = gp.LinExpr()
        for this_j in range(n_input):
            value = 1 if this_i == this_j else 0  # Diagonal is identity (1s)
            coeff = value - p_sparse[this_i, this_j]  # Use sparse matrix
            
            # Only add terms if the coefficient is larger than the tolerance
            if abs(coeff) > tolerance:
                constraint_expr.addTerms(coeff, this_z_plus[this_j])
                constraint_expr.addTerms(-coeff, this_z_minus[this_j])
        this_model.addConstr(constraint_expr == 0, f"constraint_{this_i}")

    # Step 6: Define the objective function
    obj_expr = gp.LinExpr()
    obj_expr.add(gp.quicksum(this_z_plus[this_i] - this_z_minus[this_i] for this_i in range(n_input)))
    this_model.setObjective(obj_expr, GRB.MAXIMIZE)

    return this_model, this_z_plus, this_z_minus, DyX, p_sparse  # Returning sparse version of p




def create_model_previous4(X_input, y_input, n_input):
    # Step 1: Perform operations on the matrices
    Dy = np.diag(y_input)
    DyX = Dy @ X_input
    
    # QR decomposition (DyX may still be large, but we avoid storing p explicitly)
    Q, R = np.linalg.qr(DyX, "reduced")
    
    # Use a constant value for the diagonal matrix instead of creating full dense matrices
    C = 1.0
    one_diagonal = np.ones(n_input)  # Diagonal of identity matrix (1s)

    # Step 2: Create Gurobi model
    this_model = gp.Model()

    # Set memory limit to avoid out-of-memory errors (adjust as necessary)
    this_model.setParam('MemLimit', 100000)  # Limit to 100GB

    # Step 3: Create variables
    this_z_plus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_plus_{this_i}", lb=1e-16) for this_i in range(n_input)]
    this_z_minus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_minus_{this_i}", lb=1e-16) for this_i in range(n_input)]

    # Step 4: Add the norm constraint
    norm_z = this_model.addVar()
    this_model.addConstr(norm_z == gp.quicksum(this_z_plus[this_i] + this_z_minus[this_i] for this_i in range(n_input)))
    this_model.addConstr(norm_z <= C)

    # Step 5: Add element-wise constraints, with on-the-fly calculation of p[this_i, this_j]
    tolerance = 1e-12  # Set a tolerance to filter out small values
    for this_i in range(n_input):
        constraint_expr = gp.LinExpr()
        for this_j in range(n_input):
            value = 1 if this_i == this_j else 0  # Diagonal is identity (1s)
            # Calculate p[this_i, this_j] on-the-fly instead of storing it
            p_value = Q[this_i, :] @ Q[:, this_j]
            coeff = value - p_value
            
            # Only add terms if the coefficient is larger than the tolerance
            if abs(coeff) > tolerance:
                constraint_expr.addTerms(coeff, this_z_plus[this_j])
                constraint_expr.addTerms(-coeff, this_z_minus[this_j])
        this_model.addConstr(constraint_expr == 0, f"constraint_{this_i}")

    # Step 6: Define the objective function
    obj_expr = gp.LinExpr()
    obj_expr.add(gp.quicksum(this_z_plus[this_i] - this_z_minus[this_i] for this_i in range(n_input)))
    this_model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Return the model and variables (DyX and Q are optional here, since p is no longer stored)
    return this_model, this_z_plus, this_z_minus, DyX, Q




def create_model_previous5(X_input, y_input, n_input):
    # Step 1: Perform operations on the matrices
    Dy = np.diag(y_input)
    DyX = Dy @ X_input
    
    # QR decomposition (DyX may still be large, but we avoid storing p explicitly)
    Q, R = np.linalg.qr(DyX, "reduced")
    
    # Use a constant value for the diagonal matrix instead of creating full dense matrices
    C = 1.0
    one_diagonal = np.ones(n_input)  # Diagonal of identity matrix (1s)

    # Step 2: Create Gurobi model
    this_model = gp.Model()
    
    num_available_threads = model.getEnv().getAttr('Threads')
    logger.info("num of threads")
    logger.info(num_available_threads)
    model.setParam('Threads', num_available_threads)

    # Set memory limit to avoid out-of-memory errors (adjust as necessary)
    this_model.setParam('MemLimit', 100000)  # Limit to 100GB

    # Step 3: Create variables
    this_z_plus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_plus_{this_i}", lb=1e-16) for this_i in range(n_input)]
    this_z_minus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_minus_{this_i}", lb=1e-16) for this_i in range(n_input)]

    # Step 4: Add the norm constraint
    norm_z = this_model.addVar()
    this_model.addConstr(norm_z == gp.quicksum(this_z_plus[this_i] + this_z_minus[this_i] for this_i in range(n_input)))
    this_model.addConstr(norm_z <= C)

    # Step 5: Add element-wise constraints, with on-the-fly calculation of p[this_i, this_j]
    tolerance = 1e-12  # Set a tolerance to filter out small values
    for this_i in range(n_input):
        constraint_expr = gp.LinExpr()
        for this_j in range(n_input):
            value = 1 if this_i == this_j else 0  # Diagonal is identity (1s)
            # Calculate p[this_i, this_j] on-the-fly using dot product
            p_value = np.dot(Q[this_i, :], Q[:, this_j])
            coeff = value - p_value
            
            # Only add terms if the coefficient is larger than the tolerance
            if abs(coeff) > tolerance:
                constraint_expr.addTerms(coeff, this_z_plus[this_j])
                constraint_expr.addTerms(-coeff, this_z_minus[this_j])
        this_model.addConstr(constraint_expr == 0, f"constraint_{this_i}")

    # Step 6: Define the objective function
    obj_expr = gp.LinExpr()
    obj_expr.add(gp.quicksum(this_z_plus[this_i] - this_z_minus[this_i] for this_i in range(n_input)))
    this_model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Return the model and variables (DyX and Q are optional here, since p is no longer stored)
    return this_model, this_z_plus, this_z_minus, DyX, Q




def create_model(X_input, y_input, n_input):
    # Step 1: Perform operations on the matrices
    Dy = np.diag(y_input)
    DyX = Dy @ X_input
    
    # QR decomposition (DyX may still be large, but we avoid storing p explicitly)
    Q, R = np.linalg.qr(DyX, "reduced")  # Q is (n_input, k), where k is less than n_input

    k = Q.shape[1]  # The number of columns in Q after reduced QR
    
    # Use a constant value for the diagonal matrix instead of creating full dense matrices
    C = 1.0
    one_diagonal = np.ones(n_input)  # Diagonal of identity matrix (1s)

    # Step 2: Create Gurobi model
    this_model = gp.Model()

    # Set memory limit to avoid out-of-memory errors (adjust as necessary)
    this_model.setParam('MemLimit', 100000)  # Limit to 100GB

    # Step 3: Create variables
    this_z_plus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_plus_{this_i}", lb=1e-16) for this_i in range(n_input)]
    this_z_minus = [this_model.addVar(vtype=GRB.CONTINUOUS, name=f"z_minus_{this_i}", lb=1e-16) for this_i in range(n_input)]

    # Step 4: Add the norm constraint
    norm_z = this_model.addVar()
    this_model.addConstr(norm_z == gp.quicksum(this_z_plus[this_i] + this_z_minus[this_i] for this_i in range(n_input)))
    this_model.addConstr(norm_z <= C)

    # Step 5: Add element-wise constraints, with on-the-fly calculation of p[this_i, this_j]
    tolerance = 1e-12  # Set a tolerance to filter out small values
    for this_i in range(n_input):
        constraint_expr = gp.LinExpr()
        for this_j in range(n_input):
            value = 1 if this_i == this_j else 0  # Diagonal is identity (1s)
            # Calculate p[this_i, this_j] on-the-fly using dot product, considering only the first k columns
            p_value = np.dot(Q[this_i, :k], Q[this_j, :k])  # Only use the first k columns
            coeff = value - p_value
            
            # Only add terms if the coefficient is larger than the tolerance
            if abs(coeff) > tolerance:
                constraint_expr.addTerms(coeff, this_z_plus[this_j])
                constraint_expr.addTerms(-coeff, this_z_minus[this_j])
        this_model.addConstr(constraint_expr == 0, f"constraint_{this_i}")

    # Step 6: Define the objective function
    obj_expr = gp.LinExpr()
    obj_expr.add(gp.quicksum(this_z_plus[this_i] - this_z_minus[this_i] for this_i in range(n_input)))
    this_model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Return the model and variables (DyX and Q are optional here, since p is no longer stored)
    return this_model, this_z_plus, this_z_minus, DyX, Q




logger.info("begin encoding the y label")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(adata.obs["cell_type"])
logger.info("done y_encoded")

X = adata.X
y = y_encoded
n_samples = np.shape(adata.X)[0]


#these are for centering
#X_in = X.toarray() if sparse.issparse(X) else X
#scaler = StandardScaler(with_mean=True, with_std=True)  # center only
#X_centered = scaler.fit_transform(X_in)
#X = X_centered

#remove the pca and add the centering above

pca = PCA(n_components=10, svd_solver='arpack')
pca.fit(X)
data_pca = pca.transform(X)
X = data_pca

#logger.info("done PCA")

logger.info("begin creating the model")
model, z_plus, z_minus, DyX_output, p_output = create_model_previous(X,y, n_samples)
logger.info("done creating")

model.optimize()
logger.info("done optimizing")

if model.status == GRB.OPTIMAL:
    logger.info("yay we got it")
else:
    logger.info("failed")
    logger.info(model.status)


z_plus_star = [z_plus[i].x for i in range(n_samples)]
z_minus_star = [z_minus[i].x for i in range(n_samples)]
z_star = [z_plus[i].x-z_minus[i].x for i in range(n_samples)]
logger.info("done getting z values")

model.dispose()
gp.disposeDefaultEnv()
logger.info("done disposing the model")


z_star_plus = [z_star[i] for i in range(n_samples) if z_star[i] > 0]
z_star_minus = [z_star[i] for i in range(n_samples) if z_star[i] < 0]

logger.info(norm(z_star_minus, 1))
logger.info(norm(z_star_plus, 1))

mu = norm(z_star_plus, 1)/norm(z_star_minus, 1)
logger.info("got mu!!!!")
logger.info(mu)


key = "cell_type"
logger.info("about to find out whether mu=1 in the ovr situation")
labels_raw = adata.obs[key].to_numpy()
vc = adata.obs[key].value_counts()
#majority_class = [vc.idxmax()]
majority_class = np.unique(labels_raw)

def mu_for_class(Hprime, labels, target_class, beta, tol=1e-12):
    y_pm = np.where(labels == target_class, +1.0, -1.0)   # OvR ±1
    A = y_pm[:, None] * Hprime                             # D_y * H'
    Y = A @ beta
    P = float(np.sum(np.abs(Y)))
    Q = float(np.sum(Y))
    if P <= tol:
        return float("inf")
    denom = P - Q
    if abs(denom) <= tol:
        return float("inf")
    return (P + Q) / denom

beta = np.random.randn(X.shape[1]); beta /= np.linalg.norm(beta)

A = DyX_output            # = D_y @ X  (shape n × d)
z = np.asarray(z_star)    # shape n

beta_star, *_ = lstsq(A, z)
# or: beta_star = np.linalg.pinv(A) @ z


def beta_proj_orth_to_class(X, labels_raw, target_class, beta, tol=1e-12):
    """Project beta to be orthogonal to g_c = X^T y^{(c)} so that Q=0."""
    ypm = np.where(labels_raw == target_class, +1.0, -1.0).astype(float)
    g = X.T @ ypm                      # shape (d,)
    g2 = float(np.dot(g, g))
    if g2 <= tol:
        # No projection needed (already ~orthogonal or y has no energy in X)
        b = beta
    else:
        b = beta - (np.dot(beta, g) / g2) * g
    n = float(np.linalg.norm(b))
    return b if n <= tol else b / n



mus_star = {c: mu_for_class(X, labels_raw, c, beta_star) for c in majority_class}
mus = {c: mu_for_class(X, labels_raw, c, beta) for c in majority_class}


#mus_star = {c: mu_for_class(X, labels_raw, c, beta_proj_orth_to_class(X,labels_raw, c, beta_star, tol=1e-12)) for c in majority_class}
mu_star_summary = {"max": max(mus_star.values()), "median": float(np.median(list(mus_star.values()))), "mean": float(np.mean(list(mus_star.values()))), "min": min(mus_star.values())}

#mus = {c: mu_for_class(X, labels_raw, c, beta_proj_orth_to_class(X,labels_raw, c, beta, tol=1e-12)) for c in majority_class}
mu_summary = {"max": max(mus.values()), "median": float(np.median(list(mus.values()))), "mean": float(np.mean(list(mus.values()))), "min": min(mus.values())}




logger.info("mus_star when using beta_star")
logger.info(mu_star_summary)

logger.info("mus when using any beta")
logger.info(mu_summary)


