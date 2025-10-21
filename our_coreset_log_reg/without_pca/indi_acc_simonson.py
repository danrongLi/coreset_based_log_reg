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
import tables
import h5py
import psutil  # For checking system memory
from scipy.sparse import issparse
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import log_loss

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


### Step 1: Train the Logistic Regression Model ###
def train_logistic_regression(X_train, y_train):
    """
    Train logistic regression model with multi-class 'one-vs-rest' strategy.
    """
    #if X_train.shape[0]< 50000:
    #    max_iter_val = 1000
    #elif indata.shape[0] < 500000:
    #    max_iter_val = 500
    #else:
    #    max_iter_val = 200
    
    #if len(y_train)>50000:
    #    solver_val = 'saga'
    #else:
    #    solver_val = 'lbfgs'
    
    #logger.info("max_iter")
    #logger.info(max_iter_val)
    #logger.info("solver")
    #logger.info(solver_val)

    if not isinstance(X_train, csr_matrix):
        X_train = csr_matrix(X_train)
    
    logger.info("begin training")
    begin = time.time()
    clf = LogisticRegression(multi_class='ovr', solver="saga", max_iter=200, n_jobs=-1)
    clf.fit(X_train, y_train)
    end = time.time()
    logger.info("end training")
    time_taken = end-begin
    logger.info(f"Time taken for training: {time_taken:.4f} seconds")
    return clf

def not_train_logistic_regression(X_train, y_train):
    """
    Train logistic regression model with multi-class 'one-vs-rest' strategy.
    """
    logger.info("begin training")
    begin = time.time()
    #clf = LogisticRegression(C=0.001,multi_class='ovr', solver='lbfgs', max_iter=200, n_jobs=-1)
    clf = LogisticRegression(C=0.1,multi_class='ovr', solver='lbfgs', max_iter=200, n_jobs=-1)
    #clf = LogisticRegression(C=1,multi_class='ovr', solver='lbfgs', max_iter=200, n_jobs=-1)
    clf.fit(X_train, y_train)
    end = time.time()
    logger.info("end training")
    time_taken = end-begin
    logger.info(f"Time taken for training: {time_taken:.4f} seconds")
    return clf


### Step 2: Predict Cell Types with Confidence ###
def previous_predict_with_confidence(model, X_test):
    """
    Predict the labels and also calculate the decision and probability matrices.
    """
    logger.info("begin finding decision and prob matrix")
    decision_matrix = model.decision_function(X_test)
    probability_matrix = model.predict_proba(X_test)
    
    # Get the predicted labels
    y_pred = np.argmax(probability_matrix, axis=1)
    
    return y_pred, decision_matrix, probability_matrix

### Step 4: Predict Cell Types with Temporary Model Adjustment ###
def predict_with_confidence(model, X_test_scaled, lr_idx, mode='best match', p_thres=0.5):
    # Temporarily adjust the model's attributes to match filtered gene set
    original_n_features = model.n_features_in_
    original_coef = model.coef_

    # Adjust model attributes for prediction
    model.n_features_in_ = lr_idx.size
    model.coef_ = model.coef_[:, lr_idx]
    
    decision_matrix = model.decision_function(X_test_scaled)
    probability_matrix = model.predict_proba(X_test_scaled)
    
    # Restore original model attributes
    model.n_features_in_ = original_n_features
    model.coef_ = original_coef

    # Make predictions based on mode
    if mode == 'best match':
        y_pred = np.argmax(probability_matrix, axis=1)
    else:
        y_pred = np.where(probability_matrix > p_thres, 1, 0)

    return y_pred, decision_matrix, probability_matrix



### Step 3: Gene Filtering to Match Model Features ###
def previous_filter_genes(X_data, model_features, input_gene_names):
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

def need_to_remove_0_for_wistar_filter_genes(X_data, model_features, input_gene_names):
    # Identify the indices of input genes that match model features
    matching_genes = np.isin(input_gene_names, model_features)
    if matching_genes.sum() == 0:
        raise ValueError("No matching genes between input data and model features.")

    # Order genes to match the model's features
    k_x_idx = np.where(matching_genes)[0]
    filtered_gene_names = input_gene_names[matching_genes]
    lr_idx = pd.Series(model_features).reset_index().set_index(0).loc[filtered_gene_names, 'index'].values

    # Filter X_data to contain only matching genes
    X_filtered = X_data[:, k_x_idx]
    
    return X_filtered, lr_idx

def filter_genes(X_data, model_features, input_gene_names):
    # Identify the indices of input genes that match model features
    matching_genes = np.isin(input_gene_names, model_features)
    if matching_genes.sum() == 0:
        raise ValueError("No matching genes between input data and model features.")

    # Order genes to match the model's features
    k_x_idx = np.where(matching_genes)[0]
    filtered_gene_names = input_gene_names[matching_genes]

    # Convert model_features into a DataFrame with explicit naming
    model_features_df = pd.Series(model_features, name='feature_name').reset_index()
    
    # Filter using filtered_gene_names
    try:
        lr_idx = model_features_df.set_index('feature_name').loc[filtered_gene_names, 'index'].values
    except KeyError as e:
        raise KeyError(f"Some filtered_gene_names not found in model_features: {e}")

    # Filter X_data to contain only matching genes
    X_filtered = X_data[:, k_x_idx]

    return X_filtered, lr_idx


### Step 4: Scaling the Data ###
def previous_scale_data(X_data, scaler, gene_indices):
    """
    Scale the input data using a provided StandardScaler.
    """
    # Scale the data based on the provided scaler
    X_scaled = (X_data - scaler.mean_[gene_indices]) / scaler.scale_[gene_indices]
    X_scaled[X_scaled > 10] = 10  # Clip extreme values

    return X_scaled

def without_batch_scale_data(X_data, scaler, lr_idx):
    # Scale data based on the indices of matching genes
    X_scaled = (X_data - scaler.mean_[lr_idx]) / scaler.scale_[lr_idx]
    X_scaled[X_scaled > 10] = 10  # Clip extreme values
    return X_scaled


def scale_data(X_data, scaler, train_gene_indices, batch_size=10000):
    n_samples = X_data.shape[0]
    scaled_data = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X_data[start:end, :]
        batch_scaled = (batch - scaler.mean_[train_gene_indices]) / scaler.scale_[train_gene_indices]
        scaled_data.append(batch_scaled)
    return np.vstack(scaled_data)


#def scale_data(X_data, scaler,train_gene_indices, batch_size=10000):
#    logger.info("inside scale_data function")
#    n_samples = X_data.shape[0]
#    scaled_data = []
#    for start in range(0, n_samples, batch_size):
#        end = min(start + batch_size, n_samples)
#        batch = X_data[start:end, :]        
        # Apply scaler.transform to the batch (focusing on train_gene_indices)
#        batch_scaled = scaler.transform(batch)[:, train_gene_indices]
#        scaled_data.append(batch_scaled)
#    logger.info("done scale_data function")
#    return np.vstack(scaled_data)


def custom_log_loss_multiclass(X, y, betas, lambda_reg):
    """
    Calculate the logistic loss with L2 regularization for a multi-class setting.

    Parameters:
    X (np.ndarray): Feature matrix, shape (n_samples, n_features).
    y (np.ndarray): Labels, shape (n_samples,).
    betas (np.ndarray): Logistic regression coefficients, shape (n_classes, n_features).
    lambda_reg (float): Regularization parameter.

    Returns:
    float: Calculated average loss with regularization.
    """
    n_classes, n_features = betas.shape
    logger.info(f"Number of classes: {n_classes}, Number of features: {n_features}")

    n_samples = X.shape[0]  # Number of samples
    total_loss = 0.0

    # Loop over each class to calculate the class-specific loss
    for c in range(n_classes):
        #logger.info("y")
        #logger.info(y)
        # Create a binary vector for the current class (1 for current class, -1 for others)
        y_binary = (y == c).astype(int) * 2 - 1  # Convert to -1, 1
        #logger.info("y_binary")
        #logger.info(y_binary)
        # Calculate the linear combination of features and coefficients for the current class
        linear_combination = X.dot(betas[c])

        # Calculate the logistic loss part for this class using np.logaddexp for stability
        class_loss = np.sum(np.logaddexp(0, -linear_combination * y_binary))

        # Add to total loss
        total_loss += class_loss

    # Average the total loss over all samples
    average_loss = total_loss / (n_samples*n_classes)

    # Add the L2 regularization term
    regularization_term = (lambda_reg) * np.sum(betas ** 2)
    #total_loss_with_reg = average_loss + regularization_term
    logger.info("regularization term")
    logger.info(regularization_term)

    return average_loss

def without_weighted_custom_ovr_log_loss_with_scores(true_labels, predicted_probs):
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


def custom_ovr_log_loss_with_scores(true_labels, predicted_probs):
    """
    Custom one-vs-rest log loss calculation with class weights.

    Parameters:
    - true_labels: Array of true class indices.
    - predicted_probs: 2D array of predicted probabilities (shape: [n_samples, n_classes]).

    Returns:
    - Weighted average one-vs-rest log loss.
    """
    n_samples, n_classes = predicted_probs.shape
    epsilon = 1e-15  # To avoid log(0)
    clipped_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    total_loss = 0.0

    # Calculate class weights inversely proportional to class frequencies
    class_counts = np.bincount(true_labels)
    class_weights = 1.0 / (class_counts + epsilon)

    # Loop over each class to calculate the one-vs-rest log loss
    for c in range(n_classes):
        y_binary = (true_labels == c).astype(int)
        class_weight = class_weights[c] if c < len(class_weights) else 1.0
        class_loss = -np.sum(y_binary * np.log(clipped_probs[:, c]) +
                             (1 - y_binary) * np.log(1 - clipped_probs[:, c]))
        total_loss += class_weight * class_loss

    # Average over all samples and classes
    average_loss = total_loss / n_samples
    return average_loss

### Full Pipeline Workflow ###
def run_pipeline(selected_ada, X_train,X_train_entire, y_train,y_train_entire, X_test,y_test, gene_names,model_features, resolution=1.0, min_prop=0.0):
    """
    Run the full CellTypist-like pipeline.
    """
    le = LabelEncoder()
    all_labels = np.unique(np.concatenate([y_train_entire, y_test]))
    le.fit(all_labels)


    y_train_entire_encoded = le.transform(y_train_entire)
    y_train_encoded = le.transform(y_train)
    if np.isnan(y_train_encoded).any():
        logger.info("Encoded y_train contains NaN values.")


    # Initialize the scaler and fit it on the training data
    scaler = StandardScaler(with_mean=True, with_std=True)
    #X_train_scaled = scaler.fit_transform(X_train.toarray())  # Convert to dense temporarily for scaling
    #X_train_scaled = scaler.fit_transform(X_train.toarray()).astype(float)
    
    #X_train_entire = scaler.fit_transform(X_train_entire.toarray()).astype(float)
    #X_train_scaled = scaler.transform(X_train.toarray()).astype(float)
    

    #X_train_scaled = scaler.fit_transform(X_train.toarray()).astype(float)
    #X_train_entire_scaled = scaler.transform(X_train_entire.toarray()).astype(float)
    X_train_filtered, train_gene_indices = filter_genes(X_train, model_features, gene_names)
    X_train_filtered_scaled = scaler.fit_transform(X_train_filtered.toarray()).astype(float)
    X_train_entire_filtered, train_entire_gene_indices = filter_genes(X_train_entire, model_features, gene_names)
    X_train_entire_filtered_scaled = scale_data(X_train_entire_filtered.toarray(), scaler, train_gene_indices)

    # Train Logistic Regression model
    clf = not_train_logistic_regression(X_train_filtered_scaled, y_train_encoded)
    # Filter and scale the test data to match model features
    X_test_filtered, gene_indices = filter_genes(X_test, model_features, gene_names)
    X_test_scaled = scale_data(X_test_filtered.toarray(), scaler, gene_indices)


    # Predict labels and calculate decision and probability matrices
    begin_predict = time.time()
    y_pred, decision_matrix, probability_matrix = predict_with_confidence(clf, X_test_scaled, gene_indices)
    end_predict = time.time()
    logger.info(f"Time taken for prediction: {end_predict-begin_predict:.4f} seconds")
    
    lambda_reg = 1 / (2 * clf.get_params()['C'])
    beta = clf.coef_
    #l2_penalty = lambda_reg * np.sum(beta ** 2)
    #logger.info(f"l2_penalty: {l2_penalty}")

    
    #loss2 = custom_log_loss_multiclass(X_train_filtered_scaled, y_train_encoded, beta, lambda_reg)
    #logger.info(f"previous Logistic loss for tilde beta + tilde H: {loss2}")
    _, _, prob_matrix_train_sampled = predict_with_confidence(clf, X_train_filtered_scaled, train_gene_indices)
    log_loss_train_sampled = custom_ovr_log_loss_with_scores(y_train_encoded, prob_matrix_train_sampled)
    logger.info(f"now Sampled Train Set Log Loss: {log_loss_train_sampled}")




    y_test_encoded = le.transform(y_test) 
    accuracy = accuracy_score(y_test_encoded, y_pred)
    logger.info(f"Accuracy on test data: {accuracy}")
    
    class_counts = np.bincount(y_test_encoded)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_test_encoded]

    weighted_accuracy = accuracy_score(y_test_encoded, y_pred, sample_weight=sample_weights)
    logger.info(f"weighted Accuracy on test data: {weighted_accuracy}")
    

    macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')
    logger.info(f"macro_f1 score on test data: {macro_f1}")


    balanced_acc = balanced_accuracy_score(y_test_encoded, y_pred)
    logger.info(f"balabced acc on test data: {balanced_acc}")

    # Generate the classification report as a dictionary
    report = classification_report(y_test_encoded, y_pred, output_dict=True)

    # Convert the dictionary to a pandas DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    #adding classification report code to add 2 more columns
    dist_entire = y_train_entire.value_counts()
    dist_sample = y_train.value_counts()
    report_df["n_train_entire"] = 0
    report_df["n_train"] = 0
    
    for c in range(len(le.classes_)):
        class_str_index = str(c)  # classification_report keys are strings of the class integer
        if class_str_index in report_df.index:
            # get the real string label from the numeric label
            real_class_name = le.inverse_transform([c])[0]
            # look up how many times that label appears in entire vs. sample
            n_entire = dist_entire.get(real_class_name, 0)
            n_sample = dist_sample.get(real_class_name, 0)
            report_df.at[class_str_index, "n_train_entire"] = n_entire
            report_df.at[class_str_index, "n_train"] = n_sample
    #done adding the 2 columns   



    X_train_size = len(y_train)
    
    file_name = f'/storage/home/dvl5760/work/our_log_reg/change_coreset/add_loss/test_train_split/add_crossval_fix_cellnum/reports/classification_report_simonson_size_{X_train_size}_0.1.csv'

    # Save the DataFrame as a CSV file
    report_df.to_csv(file_name, index=True)

    logger.info("done saving the classification report")
    
    #now plotting the confusion matrix
    #y axis: true label. x axis: predicted
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Train size = {X_train_size})")
    plt.tight_layout()
    cm_file = (
        f"/storage/home/dvl5760/work/our_log_reg/change_coreset/add_loss/"
        f"test_train_split/add_crossval_fix_cellnum/pdf/confusion_matrix_simonson_size_{X_train_size}_0.1.pdf"
    )
    plt.savefig(cm_file)
    plt.close()
    logger.info(f"Confusion matrix saved: {cm_file}")


    
    #loss1 = custom_log_loss_multiclass(X_test_scaled, y_test_encoded, beta, lambda_reg)
    #logger.info(f"previous Logistic loss for tilde beta + H test: {loss1}")
    log_loss_value = custom_ovr_log_loss_with_scores(y_test_encoded, probability_matrix)
    logger.info(f"now Custom One-vs-Rest Log Loss: {log_loss_value}")





    #loss = custom_log_loss_multiclass(X_train_entire_filtered_scaled, y_train_entire_encoded, beta, lambda_reg)
    #logger.info(f"previous Log loss for tilde beta + H: {loss}")
    _, _, prob_matrix_train_entire = predict_with_confidence(clf, X_train_entire_filtered_scaled, train_entire_gene_indices)
    log_loss_train_entire = custom_ovr_log_loss_with_scores(y_train_entire_encoded, prob_matrix_train_entire)
    logger.info(f"now Entire Train Set Log Loss: {log_loss_train_entire}")




    #prob_matrix_train_entire = clf.predict_proba(X_train_entire)
    #log_loss_tilde_beta_H = log_loss(y_train_entire, prob_matrix_train_entire, labels = np.unique(y_train_entire))
    #logger.info(f"Log loss for tilde beta + H: {log_loss_tilde_beta_H}")

    #probability_matrix_full = clf.predict_proba(np.concatenate([X_train_scaled, X_test_scaled]))
    #y_full_encoded = np.concatenate([y_train_encoded, y_test_encoded])
    #log_loss_tilde_beta_H_test_train = log_loss(y_full_encoded, probability_matrix_full)
    #logger.info(f"Logistic loss for tilde beta + H test train: {log_loss_tilde_beta_H_test_train}")

    y_pred_str = np.where(y_pred == -1, 'Heterogeneous', y_pred)
    y_pred_str = le.inverse_transform(y_pred_str[y_pred_str != 'Heterogeneous'].astype(int))


    return y_pred_str, clf, X_test_scaled, le


#selected_ada = anndata.read("/storage/home/dvl5760/scratch/ratmap_scp.h5ad")
#logger.info("wistar rate Data")
#selected_ada = anndata.read("/storage/home/dvl5760/scratch/SG_publication_allmice_DVC.h5ad")
#logger.info("micedvc Data")
selected_ada = anndata.read_h5ad("/scratch/dvl5760/simonson_ready_for_jupyter_uniformed.h5ad")
logger.info("Simonson Data")
#selected_ada = anndata.read("/storage/home/dvl5760/scratch/ratmap_scp.h5ad")
#logger.info("wistar rat Data")
#selected_ada = anndata.read("/storage/home/dvl5760/scratch/scp-atlas-export.h5ad")
#logger.info("skin Data")

key = 'cell_type' #simonson + heca
#key = 'leiden1p2_labels' #wistar
#key = 'celltype_coarse' #skin
#key = 'identity_layer2' #micedvc


logger.info("before dropping nan, shape:")
logger.info(selected_ada.shape)
selected_ada = selected_ada[~selected_ada.obs[key].isna()]


sc.pp.normalize_total(selected_ada, target_sum=1e4)
sc.pp.log1p(selected_ada)

logger.info(selected_ada.shape)


#not in micedvc, not in heca, but inside simonson
cell_type_counts = selected_ada.obs[key].value_counts()
valid_cell_types = cell_type_counts[cell_type_counts >= 2].index
selected_ada = selected_ada[selected_ada.obs[key].isin(valid_cell_types)].copy()
#finished


cell_type_counts = selected_ada.obs[key].value_counts()
plt.figure(figsize=(10, 6))
cell_type_counts.plot(kind='bar', color='skyblue')
plt.title("Cell Type Distribution in skin Data")
plt.xlabel("Cell Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/storage/home/dvl5760/work/our_log_reg/change_coreset/add_loss/test_train_split/add_crossval_fix_cellnum/pdf/cell_type_distribution_simonson_entire_0.1.pdf", format="pdf")

#train_indices, test_indices = train_test_split(range(selected_ada.n_obs), test_size=0.2, random_state=42)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#adata_train = selected_ada[train_indices].copy()
#adata_test = selected_ada[test_indices].copy()
for train_indices, test_indices in splitter.split(selected_ada.X, selected_ada.obs[key]):
    adata_train = selected_ada[train_indices].copy()
    adata_test = selected_ada[test_indices].copy()
X_train = adata_train.X
y_train = adata_train.obs[key]
X_test = adata_test.X
y_test = adata_test.obs[key]

train_distribution = y_train.value_counts()
test_distribution = y_test.value_counts()

plt.figure(figsize=(8, 6))
train_distribution.plot(kind='bar', color='skyblue')
plt.title("Train Cell Type Distribution")
plt.xlabel("Cell Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/storage/home/dvl5760/work/our_log_reg/change_coreset/add_loss/test_train_split/add_crossval_fix_cellnum/pdf/cell_type_distribution_simonson_train_0.1.pdf", format="pdf")

plt.figure(figsize=(8, 6))
test_distribution.plot(kind='bar', color='skyblue')
plt.title("Test Cell Type Distribution")
plt.xlabel("Cell Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/storage/home/dvl5760/work/our_log_reg/change_coreset/add_loss/test_train_split/add_crossval_fix_cellnum/pdf/cell_type_distribution_simonson_test_0.1.pdf", format="pdf")




if not isinstance(X_train, csr_matrix):
    X_train = csr_matrix(X_train)

if not isinstance(X_test, csr_matrix):
    X_test = csr_matrix(X_test)

# Check for NaN values in X_train
if np.isnan(X_train.toarray()).any():
    logger.info("X_train contains NaN values.")

# Check for NaN values in y_train
if y_train.isnull().any():
    logger.info("y_train contains NaN values.")


logger.info("X_train shape")
logger.info(X_train.shape)
logger.info("X_test shape")
logger.info(X_test.shape)


coreset_ratio = [0.1,0.3,0.5,0.7,0.9,1.0]
#coreset_ratio = [0.9,1.0]
#coreset_ratio = [0.1,0.3,0.5,0.7]
#coreset_ratio = [0.1,0.3,0.5]
#coreset_ratio = [0.7,0.9,1.0]
#coreset_ratio = [0.9]
coreset_size = []
for ratio in coreset_ratio:
    size = int(ratio*(X_train.shape[0]))
    coreset_size.append(size)

num_unique_cell_types = len(cell_type_counts) 
logger.info("num_unique_cell_types")
logger.info(num_unique_cell_types)


for size in coreset_size:
    logger.info(size)
    
    num_per_cell_type = int(size/num_unique_cell_types)
    logger.info("for each cell type, we sample")
    logger.info(num_per_cell_type)

    logger.info("begin sampling")
    start = time.time()
    sampled_indices = celltypist.samples.downsample_adata(adata=adata_train, mode="each",n_cells=num_per_cell_type, by=key, random_state=42, return_index=True)
    #sampled_indices = np.random.choice(X_train.shape[0], size=size, replace=False)
    sampled_X_train = X_train[sampled_indices, :]
    sampled_y_train = y_train[sampled_indices]

    logger.info("done sampling")
    end = time.time()
    logger.info(f"AnnData shape after sampling: {sampled_X_train.shape}")
    taken = end - start
    logger.info(f"Time taken: {taken:.4f} seconds")


    if np.isnan(sampled_X_train.toarray()).any():
        logger.info("X_train contains NaN values.")
    if sampled_y_train.isnull().any():
        logger.info("y_train contains NaN values.")


    if sampled_y_train.nunique() < 2:
        raise ValueError("y_train must contain at least two unique classes.")
    
    sampled_train_distribution = sampled_y_train.value_counts()
    plt.figure(figsize=(8, 6))
    sampled_train_distribution.plot(kind='bar', color='skyblue')
    plt.title("Sampled Train Cell Type Distribution")
    plt.xlabel("Cell Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"/storage/home/dvl5760/work/our_log_reg/change_coreset/add_loss/test_train_split/add_crossval_fix_cellnum/pdf/cell_type_distribution_simonson_train_sampled_{size}_0.1.pdf", format="pdf")


    gene_names = selected_ada.var_names #test data gene names
    model_features = selected_ada.var_names #train data gene names
    y_pred, clf, X_test_scaled, le = run_pipeline(selected_ada,sampled_X_train,X_train, sampled_y_train,y_train, X_test,y_test, gene_names,model_features, resolution=0.5, min_prop=0.2)


    y_test_filtered = y_test

    if len(y_test_filtered) == len(y_pred):
        accuracy = accuracy_score(y_test_filtered, y_pred)
        logger.info("Accuracy before majority voting:")
        logger.info(accuracy)
    else:
        logger.error("y_test and refined_predictions are still not aligned in size!")




logger.info("all done")

