import logging
import os
import seaborn as sns
import celltypist
from celltypist import models
import numpy as np
import pandas as pd
from scipy.linalg import norm
import scanpy as sc
from matplotlib.lines import Line2D
import random
from sklearn.model_selection import LeaveOneGroupOut
# import tables
import h5py
import psutil  # For checking system memory
from scipy.sparse import issparse
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import time
from scipy import sparse
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



#selected_ada = anndata.read("/storage/home/dvl5760/scratch/heca_200k.h5ad")

#adata = celltypist.samples.downsample_adata(adata = adata_entire, mode = "total",n_cells=200000, by = "cell_type",random_state=42,return_index=False, balance_cell_type=True)

#logger.info("begin sampling")
#selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = 30000, by = "cell_type",random_state=42,return_index=False )
#logger.info("after sampling")

#logger.info(f"heca shape after sampling: {adata.shape}")

#logger.info(f"heca.X shape after sampling: {adata.X.shape}")
#logger.info(f"heca.obs shape after sampling: {adata.obs.shape}")

#logger.info("Done reading heca")


#selected_ada= anndata.read_h5ad("/scratch/dvl5760/simonson_ready_for_jupyter_uniformed.h5ad")

#logger.info("Done reading simonson")

#selected_ada= anndata.read_h5ad("/scratch/dvl5760/Zheng68K.h5ad")

#selected_ada = anndata.read_h5ad("/storage/home/dvl5760/scratch/mouse_10000.h5ad")
#selected_ada = anndata.read_h5ad("/storage/home/dvl5760/scratch/mouse_10000.h5ad")

#logger.info(f"AnnData shape: {selected_ada.shape}")
#logger.info(f"Simonson.X AnnData shape: {selected_ada.X.shape}")
#logger.info(f"Simonson.obs AnnData shape: {selected_ada.obs.shape}")


#selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = 500, by = "cell_type",random_state=42,return_index=False )
#selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = 20000, by = "celltype",random_state=42,return_index=False )

#selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = 100, by = "ClusterNm",random_state=42,return_index=False) 

#logger.info(f"After subsampling AnnData shape: {selected_ada.shape}")


#X = selected_ada.X

#logger.info("begin PCA")
#pca = PCA(n_components=100, svd_solver='randomized')
#pca.fit(X)
#data_pca = pca.transform(X)
#X = data_pca
#logger.info(f"After PCA AnnData shape: {X.shape}")

#X.write("/storage/home/dvl5760/scratch/heca.X_100000_PCA_100.csv")

#logger.info("done writing pca version to scratch")


### Step 1: Train the Logistic Regression Model ###
def train_logistic_regression(X_train, y_train):
    """
    Train logistic regression model with multi-class 'one-vs-rest' strategy.
    """
    logger.info("begin creating clf and training clf")
    clf = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train, y_train)
    logger.info("got clf")
    return clf

### Step 2: Predict Cell Types with Confidence ###
def predict_with_confidence(model, X_test):
    """
    Predict the labels and also calculate the decision and probability matrices.
    """
    logger.info("begin finding decision and prob matrix")
    decision_matrix = model.decision_function(X_test)
    probability_matrix = model.predict_proba(X_test)
    
    # Get the predicted labels
    logger.info("begin finding y_pred")
    y_pred = np.argmax(probability_matrix, axis=1)
    
    logger.info("done prediecting")
    return y_pred, decision_matrix, probability_matrix

### Step 3: Gene Filtering to Match Model Features ###
def filter_genes(X_data, model_features, input_gene_names):
    """
    Filter genes in the input data to match those used in the model.
    """
    # Find common genes
    matching_genes = np.isin(input_gene_names, model_features)
    
    if matching_genes.sum() == 0:
        raise ValueError("No matching genes between input data and model features.")
    
    # Filter input data and gene names
    X_filtered = X_data[:, matching_genes]
    filtered_gene_names = input_gene_names[matching_genes]
    
    # Match input genes with model genes
    model_gene_indices = pd.Series(model_features).isin(filtered_gene_names).values
    
    return X_filtered, model_gene_indices

### Step 4: Scaling the Data ###
def scale_data(X_data, scaler):
    """
    Scale the input data using a provided StandardScaler.
    """
    # Scale the data based on the provided scaler
    #X_scaled = (X_data - scaler.mean_[gene_indices]) / scaler.scale_[gene_indices]
    #X_scaled[X_scaled > 10] = 10  # Clip extreme values
    
    X_scaled = (X_data - scaler.mean_) / scaler.scale_
    X_scaled[X_scaled > 10] = 10  # Clip extreme values

    return X_scaled



### Step 5: Construct Neighborhood Graph and Perform Over-Clustering ###
def over_cluster(X_train, resolution=1.0):
    """
    Perform over-clustering on the training data using the Leiden algorithm.
    """
    adata = sc.AnnData(X_train)

    # Preprocess the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    #sc.pp.scale(adata, max_value=10)

    # PCA and neighbors graph construction
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)

    # Perform Leiden clustering
    sc.tl.leiden(adata, resolution=resolution)

    # Extract cluster assignments (over-clustering result)
    clusters = adata.obs['leiden'].astype(int).values
    return clusters

### Step 6: Apply Majority Voting with Confidence ###
#def majority_vote_with_confidence(predictions, probability_matrix, clusters, min_prop=0.0):
#    """
#    Apply majority voting within clusters, with confidence scoring.
#    """
#    refined_predictions = np.array(predictions)
    
#    for cluster in np.unique(clusters):
#        cluster_indices = np.where(clusters == cluster)[0]
#        cluster_preds = predictions[cluster_indices]

        # Find the most frequent label in this cluster
#        majority_label, majority_count = np.unique(cluster_preds, return_counts=True)
#        majority_label = majority_label[np.argmax(majority_count)]
#        majority_prop = np.max(majority_count) / len(cluster_preds)

        # Compute confidence score based on the max probability in the cluster
#        confidence_scores = probability_matrix[cluster_indices, :]
#        max_confidence_score = np.max(confidence_scores, axis=1).mean()

#        if majority_prop >= min_prop:
#            refined_predictions[cluster_indices] = majority_label
#        else:
#            refined_predictions[cluster_indices] = -1  # Handle low-confidence clusters

#    return refined_predictions



#def majority_vote_with_confidence_edit(predictions, probability_matrix, clusters, min_prop=0.0):
#    """
#    Apply majority voting within clusters, with confidence scoring.
#    """
#    refined_predictions = np.array(predictions)

#    for cluster in np.unique(clusters):
#        cluster_indices = np.where(clusters == cluster)[0]
        
        # Ensure no index exceeds the size of predictions
#        valid_cluster_indices = cluster_indices[cluster_indices < len(predictions)]

#        if len(valid_cluster_indices) == 0:
#            logger.warning(f"Cluster {cluster} has invalid indices.")
#            continue

#        cluster_preds = predictions[valid_cluster_indices]

        # Find the most frequent label in this cluster
#        majority_label, majority_count = np.unique(cluster_preds, return_counts=True)
#        majority_label = majority_label[np.argmax(majority_count)]
#        majority_prop = np.max(majority_count) / len(cluster_preds)

        # Compute confidence score based on the max probability in the cluster
#        confidence_scores = probability_matrix[valid_cluster_indices, :]
#        max_confidence_score = np.max(confidence_scores, axis=1).mean()

#        if majority_prop >= min_prop:
#            refined_predictions[valid_cluster_indices] = majority_label
#        else:
#            refined_predictions[valid_cluster_indices] = -1  # Handle low-confidence clusters

#    return refined_predictions


### Full Pipeline Workflow ###
def run_pipeline(X_train, y_train, X_test, gene_names,model_features, resolution=1.0, min_prop=0.0):
    """
    Run the full CellTypist-like pipeline.
    """
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Initialize the scaler and fit it on the training data
    scaler = StandardScaler(with_mean=True, with_std=True)
    # X_train_scaled = scaler.fit_transform(X_train)  # Convert to dense temporarily for scaling
    X_train_scaled = scaler.fit_transform(X_train.toarray())  # Convert to dense temporarily for scaling

    # Train Logistic Regression model
    logger.info("begin training")
    current=time.time()
    clf = train_logistic_regression(X_train_scaled, y_train)
    logger.info("done training")
    end=time.time()
    logger.info("training time seconds")
    logger.info(end-current)
    # Filter and scale the test data to match model features
    #X_test_filtered, gene_indices = filter_genes(X_test, model_features, gene_names)
    # X_test_scaled = scale_data(X_test, scaler)
    X_test_scaled = scale_data(X_test.toarray(), scaler)

    # Predict labels and calculate decision and probability matrices
    y_pred, decision_matrix, probability_matrix = predict_with_confidence(clf, X_test_scaled)
    
    y_pred_str = np.where(y_pred == -1, 'Heterogeneous', y_pred)
    y_pred_str = le.inverse_transform(y_pred_str[y_pred_str != 'Heterogeneous'].astype(int))

    # Perform over-clustering on the training data
    #clusters = over_cluster(X_train, resolution=resolution)

    # Apply majority voting with confidence
    #refined_predictions = majority_vote_with_confidence(y_pred, probability_matrix, clusters, min_prop=min_prop)
    
    # Convert refined_predictions back to string labels
    #refined_predictions_str = np.where(refined_predictions == -1, 'Heterogeneous', refined_predictions)
    #refined_predictions_str = le.inverse_transform(refined_predictions_str[refined_predictions_str != 'Heterogeneous'].astype(int))


    return y_pred_str, clf, X_test_scaled, le



# Step to filter y_test based on X_test filtering
def filter_y_test(y_test, X_test, refined_predictions):
    """
    Ensure that y_test matches the number of cells in refined_predictions by filtering out entries.
    """
    if len(y_test) != len(refined_predictions):
        # Check which cells were kept during filtering
        filtered_indices = np.arange(len(refined_predictions))  # Assuming the indices of refined_predictions match filtered cells

        # Apply the same filtering to y_test
        y_test_filtered = y_test.iloc[filtered_indices]  # Use iloc if y_test is a pandas series
    else:
        y_test_filtered = y_test
    
    return y_test_filtered


# Ensure the gene selection between HECA and Simonson datasets is aligned
def select_common_genes(adata_heca, adata_simonson):
    """
    Select common genes between HECA and Simonson datasets.
    """
    # Find the intersection of the gene names
    common_genes = np.intersect1d(adata_heca.var_names, adata_simonson.var_names)
    
    if len(common_genes) == 0:
        raise ValueError("No common genes found between HECA and Simonson datasets.")
    
    # Filter both datasets to only include common genes
    adata_heca = adata_heca[:, common_genes]
    adata_simonson = adata_simonson[:, common_genes]
    
    logger.info(f"Number of common genes: {len(common_genes)}")
    
    return adata_heca, adata_simonson


#sc.pp.normalize_total(selected_ada, target_sum=1e4)
#sc.pp.log1p(selected_ada)
#adata, selected_ada = select_common_genes(adata, selected_ada)


#train_indices, test_indices = train_test_split(range(adata.n_obs), test_size=0.2, random_state=42)
#adata_train = adata[train_indices].copy()
#adata_test = adata[test_indices].copy()




#coreset_sizes = [10000,30000,50000,100000,150000,200000]
#coreset_sizes = [500,1000,3000,5000,10000,30000]
#coreset_sizes = [1000,3000,5000,10000,30000,68450]
# coreset_sizes = [300,500,1000,3000,5000]
coreset_sizes = [300,500,1000,2000,3000,4000,5000]

for size in coreset_sizes:
    logger.info(size)
    #selected_ada = anndata.read("/storage/home/dvl5760/scratch/heca_200k.h5ad")
    #selected_ada= anndata.read_h5ad("/scratch/dvl5760/simonson_ready_for_jupyter_uniformed.h5ad")
    #selected_ada= anndata.read_h5ad("/scratch/dvl5760/Zheng68K.h5ad")
    # selected_ada = anndata.read_h5ad("/storage/home/dvl5760/scratch/mouse_10000.h5ad")
    selected_ada = anndata.read_h5ad("/Users/danrongli/Desktop/Feature_Space_Logistic/server_results/oct23/mouse_10000.h5ad")
    if size == selected_ada.shape[0]:
        selected_ada = selected_ada
    else:
        selected_ada = celltypist.samples.downsample_adata(adata = selected_ada, mode = "total",n_cells = size, by = "ClusterNm",random_state=42,return_index=False)
    
    X = selected_ada.X
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)
    
    # current = time.time()
    # svd = TruncatedSVD(n_components=100, algorithm='randomized', random_state=42)
    # data_pca = svd.fit_transform(X)
    # end = time.time()
    # logger.info("pca used seconds")
    # logger.info(end-current)
    # X = data_pca

    train_indices, test_indices = train_test_split(range(selected_ada.n_obs), test_size=0.2, random_state=42)
    adata_train = selected_ada[train_indices].copy()
    adata_test = selected_ada[test_indices].copy()

    X_train = X[train_indices]
    y_train = adata_train.obs["ClusterNm"]
    X_test = X[test_indices]
    y_test = adata_test.obs["ClusterNm"]
    #y_test = adata_test.obs["ClusterNm"]

#logger.info("train X shape")
#logger.info(X_train.shape)
#logger.info("test X shape")
#logger.info(X_test.shape)

#X_train = adata.X 
#y_train = adata.obs["cell_type"]
#X_test = selected_ada.X
#y_test = selected_ada.obs["cell_type"]


    #if not isinstance(X_train, csr_matrix):
    #    X_train = csr_matrix(X_train)

    #if not isinstance(X_test, csr_matrix):
    #    X_test = csr_matrix(X_test)


#gene_names = selected_ada.var_names #test data gene names
#model_features = adata.var_names #train data gene names

    gene_names = adata_test.var_names #test data gene names
    model_features = adata_train.var_names #train data gene names
    y_pred, clf, X_test_scaled, le = run_pipeline(X_train, y_train, X_test, gene_names,model_features, resolution=0.5, min_prop=0.2)




    y_test_filtered = filter_y_test(y_test, X_test, y_pred)
# Check sizes to ensure they are now aligned
#logger.info(f"Number of samples in y_test_filtered: {len(y_test_filtered)}")
#logger.info(f"Number of samples in y_pred: {len(y_pred)}")

    if len(y_test_filtered) == len(y_pred):
        # Calculate accuracy
        accuracy = accuracy_score(y_test_filtered, y_pred)
        logger.info("Final Accuracy:")
        logger.info(accuracy)
    else:
        logger.error("y_test and refined_predictions are still not aligned in size!")



# Optionally, print the confusion matrix
#conf_matrix = confusion_matrix(y_test_filtered, refined_predictions)
#logger.info("confusion matrix after converting -1 to hetero")
#logger.info(conf_matrix)

    #combined_df = pd.DataFrame({'True_Labels': y_test_filtered, 'Predicted_Labels': y_pred})
    #combined_df.to_csv('/storage/home/dvl5760/work/our_log_reg/mouse_100_pca_100.csv', index=False)


#unique, counts = np.unique(refined_predictions, return_counts=True)
#logger.info("Label distribution in predictions:")
#logger.info(dict(zip(unique, counts)))





logger.info("all done")

