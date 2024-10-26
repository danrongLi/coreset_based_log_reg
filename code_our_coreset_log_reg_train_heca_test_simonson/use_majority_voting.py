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
from sklearn.metrics import accuracy_score

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


#logger.info("this time, let us use celltypist to select 1000 samples per label")

#adata = anndata.read("/storage/home/dvl5760/scratch/heca_200k.h5ad")

#coreset_sizes = [1000,5000,10000,30000,50000,100000]

#for size in coreset_sizes:
#    logger.info(size)

#    adata = celltypist.samples.downsample_adata(adata = adata, mode = "total",n_cells=size, by = "cell_type",random_state=42,return_index=False)


    #adata = celltypist.samples.downsample_adata(adata = adata, mode = "each",n_cells = 100, by = "cell_type",random_state=42,return_index=False )


#    logger.info(f"heca shape after sampling: {adata.shape}")

    #logger.info(f"heca.X shape after sampling: {adata.X.shape}")
    #logger.info(f"heca.obs shape after sampling: {adata.obs.shape}")

    #logger.info("Done reading heca")


#    selected_ada= anndata.read_h5ad("/scratch/dvl5760/simonson_ready_for_jupyter_uniformed.h5ad")

    #logger.info("Done reading simonson")


#    logger.info(f"Simonson AnnData shape: {selected_ada.shape}")
    #logger.info(f"Simonson.X AnnData shape: {selected_ada.X.shape}")
    #logger.info(f"Simonson.obs AnnData shape: {selected_ada.obs.shape}")



#highly_variable_genes = adata.var_names
#common_genes = list(set(highly_variable_genes).intersection(set(selected_ada.var_names)))
#logger.info("common_genes")
#logger.info(common_genes)
#selected_ada = selected_ada[:, common_genes]

#logger.info(f"After selecting genes Simonson AnnData shape: {selected_ada.shape}")
#logger.info(f"After selecting genes Simonson.X AnnData shape: {selected_ada.X.shape}")
#logger.info(f"After selecting genes Simonson.obs AnnData shape: {selected_ada.obs.shape}")



# Normalize the total counts per cell to 10,000
#sc.pp.normalize_total(selected_ada, target_sum=1e4)
#logger.info("done normalizing total counts")
# Log1p transform the data
#sc.pp.log1p(selected_ada)
#logger.info("done log1p transform")

#logger.info("Done transforming for simonson")


#selected_ada.write("/scratch/dvl5760/after_encoder_before_embedding_jupyter_simonson.h5ad")
#logger.info("done saving the file to feed into scMulan jupyter")




### Step 1: Train the Logistic Regression Model ###
#def train_logistic_regression(X_train, y_train):
#    """
#    Train logistic regression model with multi-class 'one-vs-rest' strategy.
#    """
#    logger.info("begin creating clf and training clf")
#    clf = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000, class_weight='balanced', n_jobs=-1)
#    clf.fit(X_train, y_train)
#    logger.info("got clf")
#    return clf

def dynamic_logistic_regression(X_train, y_train, C=1.0, solver=None, max_iter=None, n_jobs=-1):
    """
    Dynamically adjust the parameters of the logistic regression classifier based on dataset size.
    
    Parameters:
    - C: Regularization strength for logistic regression.
    - solver: Solver to use for optimization ('lbfgs', 'sag', etc.).
    - max_iter: Maximum iterations for optimization.
    - n_jobs: Number of CPU cores to use (-1 means all cores).
    """
    logger.info("inisde logisitic regression function")
    # Determine dataset size
    n_samples, n_features = X_train.shape

    # Adjust `max_iter` based on the number of cells
    if max_iter is None:
        if n_samples < 50000:
            max_iter = 1000
        elif n_samples < 500000:
            max_iter = 500
        else:
            max_iter = 200
            
    # Determine solver based on the number of samples
    if solver is None:
        solver = 'sag' if n_samples > 50000 else 'lbfgs'

    # Train Logistic Regression model with dynamic parameters
    #logger.info(f"ðŸ‹ï¸ Using logistic regression with solver: {solver}, max_iter: {max_iter}")
    clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter, multi_class='ovr', n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    logger.info("exit logistic regression regression function")
    return clf


### Step 2: Predict Cell Types with Confidence ###
#def predict_with_confidence(model, X_test):
#    """
#    Predict the labels and also calculate the decision and probability matrices.
#    """
#    logger.info("begin finding decision and prob matrix")
#    decision_matrix = model.decision_function(X_test)
#    probability_matrix = model.predict_proba(X_test)
    
#    # Get the predicted labels
#    logger.info("begin finding y_pred")
#    y_pred = np.argmax(probability_matrix, axis=1)
#    
#    logger.info("done prediecting")
#    return y_pred, decision_matrix, probability_matrix


### Step 2: Predict Cell Types with Confidence ###
def predict_with_confidence(model, X_test):
    """
    Predict the labels and also calculate the decision and probability matrices.
    """
    logger.info("inside predicting function")
    decision_matrix = model.decision_function(X_test)
    probability_matrix = model.predict_proba(X_test)
    
    # Get the predicted labels
    y_pred = np.argmax(probability_matrix, axis=1)
    
    logger.info("exit predicting function")
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
#def scale_data(X_data, scaler, gene_indices):
#    """
#    Scale the input data using a provided StandardScaler.
#    """
#    # Scale the data based on the provided scaler
#    X_scaled = (X_data - scaler.mean_[gene_indices]) / scaler.scale_[gene_indices]
#    X_scaled[X_scaled > 10] = 10  # Clip extreme values
#
#    return X_scaled


#def scale_data(X_data, scaler):
#    """
#    Scale the input data using a provided StandardScaler.
#    Scales only the matching features from the logistic model.
#    """
#    logger.info("inside scaling data function")
#    # Scale only the matching gene indices
#    means_ = scaler.mean_ if scaler.with_mean else 0
#    scales_ = scaler.scale_
#
#    # Apply scaling to input data
#    X_scaled = (X_data - means_) / scales_
#    X_scaled[X_scaled > 10] = 10  # Clip extreme values
#    logger.info("exit scaling data function")
#    return X_scaled 


def scale_data(X_data, scaler, gene_indices):
    """
    Scale the input data using a provided StandardScaler.
    Scales only the matching features from the logistic model.
    """
    logger.info("inside scaling data function")
    # Scale only the matching gene indices
    means_ = scaler.mean_[gene_indices] if scaler.with_mean else 0
    scales_ = scaler.scale_[gene_indices]

    # Apply scaling to input data
    X_scaled = (X_data[:, gene_indices] - means_) / scales_
    X_scaled[X_scaled > 10] = 10  # Clip extreme values
    logger.info("exit scaling data function")
    return X_scaled


### Step 5: Construct Neighborhood Graph and Perform Over-Clustering ###
#def over_cluster(X_train, resolution=1.0):
#    """
#    Perform over-clustering on the training data using the Leiden algorithm.
#    """
#    adata = sc.AnnData(X_train)

#    # Preprocess the data
#    sc.pp.normalize_total(adata, target_sum=1e4)
#    sc.pp.log1p(adata)
#    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
#    adata = adata[:, adata.var.highly_variable]
#    #sc.pp.scale(adata, max_value=10)
#
#    # PCA and neighbors graph construction
#    sc.tl.pca(adata, n_comps=50)
#    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
#
#    # Perform Leiden clustering
#    sc.tl.leiden(adata, resolution=resolution)
#
#    # Extract cluster assignments (over-clustering result)
#    clusters = adata.obs['leiden'].astype(int).values
#    return clusters


def over_cluster(this_adata, X_test, use_GPU=False):
    """
    Perform over-clustering on the filtered test data using the Leiden algorithm with dynamic resolution.
    The input this_adata is filtered to match the test data (X_test).
    """
    logger.info("inside neighborhood graph construction and over clustering function")
    # Filter `this_adata` to match the dimensions of X_test
    filtered_adata = this_adata[:X_test.shape[0], :]  # Filter the AnnData object to match the test data

    # Adjust the resolution dynamically based on dataset size
    n_obs = filtered_adata.n_obs
    if n_obs < 5000:
        resolution = 5
    elif n_obs < 20000:
        resolution = 10
    elif n_obs < 40000:
        resolution = 15
    elif n_obs < 100000:
        resolution = 20
    elif n_obs < 200000:
        resolution = 25
    else:
        resolution = 30

    logger.info(f"Using dynamic resolution: {resolution} for over-clustering on test data")

    # Perform PCA and construct the neighborhood graph on filtered test data
    sc.tl.pca(filtered_adata, n_comps=50)
    sc.pp.neighbors(filtered_adata, n_neighbors=10, n_pcs=50)

    # Perform Leiden clustering
    sc.tl.leiden(filtered_adata, resolution=resolution)

    clusters = filtered_adata.obs['leiden'].astype(int).values
    logger.info("exit neighborhood graph construction and over clustering function")
    return clusters




### Step 6: Apply Majority Voting with Confidence ###
#def majority_vote_with_confidence(predictions, probability_matrix, clusters, min_prop=0.0):
#    """
#    Apply majority voting within clusters, with confidence scoring.
#    """
#    refined_predictions = np.array(predictions)
#    
#    for cluster in np.unique(clusters):
#        cluster_indices = np.where(clusters == cluster)[0]
#        cluster_preds = predictions[cluster_indices]
#
#        # Find the most frequent label in this cluster
#        majority_label, majority_count = np.unique(cluster_preds, return_counts=True)
#        majority_label = majority_label[np.argmax(majority_count)]
#        majority_prop = np.max(majority_count) / len(cluster_preds)
#        # Compute confidence score based on the max probability in the cluster
#        confidence_scores = probability_matrix[cluster_indices, :]
#        max_confidence_score = np.max(confidence_scores, axis=1).mean()
#
#        if majority_prop >= min_prop:
#            refined_predictions[cluster_indices] = majority_label
#        else:
#            refined_predictions[cluster_indices] = -1  # Handle low-confidence clusters
#
#    return refined_predictions



def majority_vote(predictions, over_clustering, probability_matrix, min_prop=0.0):
    """
    Perform majority voting based on the CellTypist methodology.

    Parameters:
    - predictions: The predicted labels for each cell (1D array).
    - over_clustering: The subcluster assignment for each cell (1D array, e.g., from Leiden clustering).
    - probability_matrix: The matrix of probabilities for each cell belonging to each class.
    - min_prop: The minimum proportion required for majority voting to confidently assign a label.

    Returns:
    - Updated predictions with majority voting applied.
    """
    logger.info("inside majority voting function")
    # Ensure predictions and over_clustering are 1D arrays
    predictions = np.ravel(predictions)
    over_clustering = np.ravel(over_clustering)

    logger.info(f"Size of predictions (y_pred): {len(predictions)}")
    logger.info(f"Size of clusters (over_clustering): {len(over_clustering)}")

    # Check that both arrays are of the same length
    if predictions.shape[0] != over_clustering.shape[0]:
        raise ValueError("Predictions and over_clustering must have the same number of elements.")

    # Create a DataFrame to facilitate voting
    df_predictions = pd.DataFrame({'predicted_labels': predictions, 'over_clustering': over_clustering})

    # Cross-tabulate the predicted labels against the subclusters
    votes = pd.crosstab(df_predictions['predicted_labels'], df_predictions['over_clustering'])

    # Identify the majority label in each subcluster
    majority = votes.idxmax(axis=0).astype(str)

    # Calculate the frequency of the majority label within each subcluster
    freqs = (votes / votes.sum(axis=0).values).max(axis=0)

    # Apply the min_prop threshold: if the frequency is below the threshold, label the subcluster as 'Heterogeneous'
    majority[freqs < min_prop] = 'Heterogeneous'

    # Assign the majority-voted label to each cell based on the over_clustering
    df_predictions['majority_voting'] = majority[df_predictions['over_clustering']].values

    logger.info("âœ… Majority voting completed!")
    logger.info("exit majority voting function")
    # df_predictions.to_csv("/storage/home/dvl5760/work/new_simonson/df_predictions.csv", index=False)
    # votes.to_csv("/storage/home/dvl5760/work/new_simonson/votes.csv", index=False)
    # majority.to_csv("/storage/home/dvl5760/work/new_simonson/majority.csv", index=False)

    # Return the updated predictions with majority voting applied
    return df_predictions['majority_voting'].values


#def majority_vote_with_confidence_edit(predictions, probability_matrix, clusters, min_prop=0.0):
#    """
#    Apply majority voting within clusters, with confidence scoring.
#    """
#   refined_predictions = np.array(predictions)
#
#    for cluster in np.unique(clusters):
#        cluster_indices = np.where(clusters == cluster)[0]
#        
#        # Ensure no index exceeds the size of predictions
#        valid_cluster_indices = cluster_indices[cluster_indices < len(predictions)]
#
#        if len(valid_cluster_indices) == 0:
#            logger.warning(f"Cluster {cluster} has invalid indices.")
#            continue
#
#        cluster_preds = predictions[valid_cluster_indices]
#
#        # Find the most frequent label in this cluster
#        majority_label, majority_count = np.unique(cluster_preds, return_counts=True)
#        majority_label = majority_label[np.argmax(majority_count)]
#        majority_prop = np.max(majority_count) / len(cluster_preds)
#
#        # Compute confidence score based on the max probability in the cluster
#        confidence_scores = probability_matrix[valid_cluster_indices, :]
#        max_confidence_score = np.max(confidence_scores, axis=1).mean()
#
#        if majority_prop >= min_prop:
#            refined_predictions[valid_cluster_indices] = majority_label
#        else:
#            refined_predictions[valid_cluster_indices] = -1  # Handle low-confidence clusters
#
#    return refined_predictions




### Step 7: Saving and Loading the Model ###
#def save_model(model, filepath, metadata=None):
#    """
#    Save the trained model along with metadata.
#    """
#    data = {
#        'model': model,
#        'metadata': metadata or {'date': str(datetime.now())}
#    }
#    with open(filepath, 'wb') as f:
#        pickle.dump(data, f)

#def load_model(filepath):
#    """
#    Load the trained model from file.
#    """
#    with open(filepath, 'rb') as f:
#        data = pickle.load(f)
#    return data['model'], data['metadata']

### Full Pipeline Workflow ###
#def run_pipeline(X_train, y_train, X_test, gene_names,model_features, resolution=1.0, min_prop=0.0):
#    """
#    Run the full CellTypist-like pipeline.
#    """
#    le = LabelEncoder()
#    y_train_encoded = le.fit_transform(y_train)
#
#    # Initialize the scaler and fit it on the training data
#    scaler = StandardScaler(with_mean=True, with_std=True)
#    X_train_scaled = scaler.fit_transform(X_train.toarray())  # Convert to dense temporarily for scaling
#
#    # Train Logistic Regression model
#    start = time.time()
#    clf = train_logistic_regression(X_train_scaled, y_train)
#    end = time.time()
#    logger.info("training seconds used")
#    logger.info(end-start)
#
#    # Filter and scale the test data to match model features
#    X_test_filtered, gene_indices = filter_genes(X_test, model_features, gene_names)
#    X_test_scaled = scale_data(X_test_filtered.toarray(), scaler, gene_indices)
#
#    # Predict labels and calculate decision and probability matrices
#    y_pred, decision_matrix, probability_matrix = predict_with_confidence(clf, X_test_scaled)
#
#    # Perform over-clustering on the training data
#    clusters = over_cluster(X_train, resolution=resolution)
#
#    # Apply majority voting with confidence
#    refined_predictions = majority_vote_with_confidence(y_pred, probability_matrix, clusters, min_prop=min_prop)
#    
#    # Convert refined_predictions back to string labels
#    refined_predictions_str = np.where(refined_predictions == -1, 'Heterogeneous', refined_predictions)
#    refined_predictions_str = le.inverse_transform(refined_predictions_str[refined_predictions_str != 'Heterogeneous'].astype(int))
#    
#
#    #y_pred_str = np.where(y_pred == -1, 'Heterogeneous', y_pred)
#    #y_pred_str = le.inverse_transform(y_pred_str[y_pred_str != 'Heterogeneous'].astype(int))
#
#   return refined_predictions_str, clf, X_test_scaled, le


def run_pipeline(this_adata, X_train, y_train, X_test, gene_names, model_features, use_GPU=False, min_prop=0.0):
    """
    Run the full CellTypist-like pipeline with improved scaling, clustering, and voting.
    """
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Initialize the scaler and fit it on the training data
    scaler = StandardScaler(with_mean=True)
    X_train_scaled = scaler.fit_transform(X_train.toarray())  # Convert to dense temporarily for scaling

    # Train Logistic Regression model
    start = time.time()
    clf = dynamic_logistic_regression(X_train_scaled, y_train_encoded)
    end = time.time()
    logger.info("seconds used")
    logger.info(end-start)
    # Filter and scale the test data to match model features
    X_test_filtered, gene_indices = filter_genes(X_test, model_features, gene_names)
    X_test_scaled = scale_data(X_test_filtered.toarray(), scaler, gene_indices)

    # Predict labels and calculate decision and probability matrices
    y_pred, decision_matrix, probability_matrix = predict_with_confidence(clf, X_test_scaled)

    # **Fix**: Perform over-clustering only on the filtered test data that matches X_test
    clusters = over_cluster(this_adata, X_test_scaled, use_GPU=use_GPU)

    # Apply majority voting with confidence
    refined_predictions = majority_vote(y_pred, clusters, probability_matrix, min_prop=min_prop)

    # Convert refined_predictions back to string labels
    refined_predictions_str = np.where(refined_predictions == -1, 'Heterogeneous', refined_predictions)
    refined_predictions_str = le.inverse_transform(refined_predictions_str[refined_predictions_str != 'Heterogeneous'].astype(int))

    return refined_predictions_str, clf, X_test_scaled, le, y_pred



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



#adata = anndata.read("/storage/home/dvl5760/scratch/heca_200k.h5ad")
selected_ada= anndata.read_h5ad("/Users/danrongli/Desktop/Feature_Space_Logistic/server_results/oct23/simonson_ready_for_jupyter_uniformed.h5ad")
logger.info(f"Simonson AnnData shape: {selected_ada.shape}")

#coreset_sizes = [10000, 30000, 50000, 1000, 5000, 3000, 100, 300, 500]


#coreset_sizes = [100000,150000]

#coreset_sizes = [100,300,500,1000,3000,5000,10000]

#coreset_sizes = [2000,4000,6000,8000]

coreset_sizes = [500,1000,2000,3000,4000,5000,6000,8000,10000]

for size in coreset_sizes:
    logger.info(size)
    adata = anndata.read("/Users/danrongli/Desktop/Feature_Space_Logistic/server_results/oct23/heca_200k.h5ad")
    adata = celltypist.samples.downsample_adata(adata = adata, mode = "total",n_cells=size, by = "cell_type",random_state=42,return_index=False)
    #adata = celltypist.samples.downsample_adata(adata = adata, mode = "each",n_cells = 100, by = "cell_type",random_state=42,return_index=False )
    logger.info(f"heca shape after sampling: {adata.shape}")

    adata, selected_ada = select_common_genes(adata, selected_ada)

    sc.pp.normalize_total(selected_ada, target_sum=1e4)
    sc.pp.log1p(selected_ada)
   
    #adata, selected_ada = select_common_genes(adata, selected_ada)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X_train = adata.X 
    y_train = adata.obs["cell_type"]
    X_test = selected_ada.X
    y_test = selected_ada.obs["cell_type"]

    if not isinstance(X_train, csr_matrix):
        X_train = csr_matrix(X_train)

    if not isinstance(X_test, csr_matrix):
        X_test = csr_matrix(X_test)


    gene_names = selected_ada.var_names #test data gene names
    model_features = adata.var_names #train data gene names
    #refined_predictions, clf, X_test_scaled, le = run_pipeline(X_train, y_train, X_test, gene_names,model_features, resolution=20, min_prop=0.2)
    refined_predictions, clf, X_test_scaled, le, y_pred = run_pipeline(selected_ada, X_train, y_train, X_test, gene_names,model_features, min_prop=0)



    y_test_filtered = filter_y_test(y_test, X_test, refined_predictions)
    # Check sizes to ensure they are now aligned
    #logger.info(f"Number of samples in y_test_filtered: {len(y_test_filtered)}")
    #logger.info(f"Number of samples in refined_predictions: {len(refined_predictions)}")
    

    y_pred = le.inverse_transform(y_pred[y_pred != 'Heterogeneous'].astype(int))
    y_pred_accuracy = accuracy_score(y_test_filtered, y_pred)
    logger.info("Accuracy before Majority Voting, before hierarchy:")
    logger.info(y_pred_accuracy)


    if len(y_test_filtered) == len(refined_predictions):
        # Calculate accuracy
        accuracy = accuracy_score(y_test_filtered, refined_predictions)
        logger.info("after majority voting, before hierarchy, acc:")
        logger.info(accuracy)
    else:
        logger.error("y_test and refined_predictions are still not aligned in size!")



# Optionally, print the confusion matrix
#conf_matrix = confusion_matrix(y_test_filtered, refined_predictions)
#logger.info("confusion matrix after converting -1 to hetero")
#logger.info(conf_matrix)

    combined_df = pd.DataFrame({'True_Labels': y_test_filtered,"Predicted_Labels": y_pred ,'Majority_Voting': refined_predictions})
    #combined_df.to_csv('/storage/home/dvl5760/work/new_simonson/y_test_and_predictions_100_per_celltype_original.csv', index=False)
    
    combined_df.to_csv(f'/storage/home/dvl5760/work/our_log_reg/train_heca_test_simonson/combined_majority_voting_coreset_size_{size}.csv', index=False)

#unique, counts = np.unique(refined_predictions, return_counts=True)
#logger.info("Label distribution in predictions:")
#logger.info(dict(zip(unique, counts)))

    combined_df.loc[((combined_df["Predicted_Labels"] == "Basement membrane fibroblast") | (combined_df["Predicted_Labels"] == "Myofibroblast") | (combined_df["Predicted_Labels"] == "Alveolar fibroblast")| (combined_df["Predicted_Labels"] == "Adventitial fibroblast")) & (combined_df["True_Labels"] == "Fibroblast"), "Predicted_Labels"] = "Fibroblast"

    combined_df.loc[((combined_df["Predicted_Labels"] == "Capillary endothelial cell")| (combined_df["Predicted_Labels"] == "Lymphatic endothelial cell") | (combined_df["Predicted_Labels"] == "Vascular endothelial cell")|(combined_df["Predicted_Labels"] == "Sinusoidal endothelial cell")) & (combined_df["True_Labels"] == "Endothelial cell"), "Predicted_Labels"] = "Endothelial cell"

    combined_df.loc[((combined_df["Predicted_Labels"] == "Memory CD4 T cell") | (combined_df["Predicted_Labels"] == "T cell")|(combined_df["Predicted_Labels"] == "Naive CD8 T cell")) & (combined_df["True_Labels"] == "Lymphoid cell"), "Predicted_Labels"] = "Lymphoid cell"

    combined_df.loc[(combined_df["Predicted_Labels"] == "Vascular smooth muscle cell") & (combined_df["True_Labels"] == "Smooth muscle cell"), "Predicted_Labels"] = "Smooth muscle cell"

    after_acc = accuracy_score(combined_df["Predicted_Labels"].values, combined_df["True_Labels"].values)

    logger.info("Before majority voting, after hierarchy")
    logger.info(after_acc)


    combined_df.loc[((combined_df["Majority_Voting"] == "Basement membrane fibroblast") | (combined_df["Majority_Voting"] == "Myofibroblast") | (combined_df["Majority_Voting"] == "Alveolar fibroblast")| (combined_df["Majority_Voting"] == "Adventitial fibroblast")) & (combined_df["True_Labels"] == "Fibroblast"), "Majority_Voting"] = "Fibroblast"

    combined_df.loc[((combined_df["Majority_Voting"] == "Capillary endothelial cell")| (combined_df["Majority_Voting"] == "Lymphatic endothelial cell") | (combined_df["Majority_Voting"] == "Vascular endothelial cell")|(combined_df["Majority_Voting"] == "Sinusoidal endothelial cell")) & (combined_df["True_Labels"] == "Endothelial cell"), "Majority_Voting"] = "Endothelial cell"

    combined_df.loc[((combined_df["Majority_Voting"] == "Memory CD4 T cell") | (combined_df["Majority_Voting"] == "T cell")|(combined_df["Majority_Voting"] == "Naive CD8 T cell")) & (combined_df["True_Labels"] == "Lymphoid cell"), "Majority_Voting"] = "Lymphoid cell"

    combined_df.loc[(combined_df["Majority_Voting"] == "Vascular smooth muscle cell") & (combined_df["True_Labels"] == "Smooth muscle cell"), "Majority_Voting"] = "Smooth muscle cell"

    mj_after_acc = accuracy_score(combined_df["Majority_Voting"].values, combined_df["True_Labels"].values)

    logger.info("after majority voting, after hierarchy")
    logger.info(mj_after_acc)


logger.info("all done")

