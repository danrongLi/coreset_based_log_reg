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
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold

from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import diags
from scipy.linalg import svd, lstsq
from scipy.sparse.linalg import inv, LinearOperator, spsolve, factorized

import anndata
from anndata import AnnData
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



#selected_ada = anndata.read("/storage/home/dvl5760/scratch/ratmap_scp.h5ad")
#logger.info("wistar rate Data")
#selected_ada = anndata.read("/storage/home/dvl5760/scratch/SG_publication_allmice_DVC.h5ad")
#logger.info("micedvc Data")
selected_ada = anndata.read_h5ad("/scratch/dvl5760/simonson_ready_for_jupyter_uniformed.h5ad")
logger.info("Simonson Data")
#selected_ada = anndata.read("/storage/home/dvl5760/scratch/heca_200k.h5ad")
#logger.info("hECA Data")
#selected_ada = anndata.read("/storage/home/dvl5760/scratch/scp-atlas-export.h5ad")
#logger.info("skin Data")

key = 'cell_type' #simonson + heca
#key = 'leiden1p2_labels' #wistar
#key = 'celltype_coarse' #skin
#key = 'identity_layer2' #micedvc

sc.pp.normalize_total(selected_ada, target_sum=1e4)
logger.info("done normalizing total counts")
sc.pp.log1p(selected_ada)
logger.info("done log1p transform")


#add the same splitting code
logger.info(selected_ada.shape)
cell_type_counts = selected_ada.obs[key].value_counts()
valid_cell_types = cell_type_counts[cell_type_counts >= 2].index
selected_ada = selected_ada[selected_ada.obs[key].isin(valid_cell_types)].copy()
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_indices, test_indices in splitter.split(selected_ada.X, selected_ada.obs[key]):
    adata_train = selected_ada[train_indices].copy()
    adata_test = selected_ada[test_indices].copy()
#done adding




batch_size=1000
logger.info("batch_size")
logger.info(batch_size)

logger.info("begin 3 fold cross validation")
start_cv = time.time()

C_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]


kf = KFold(n_splits=3, shuffle=True, random_state=42)

best_score_log_loss = np.inf
best_score_acc = -np.inf
best_score_bal_acc = -np.inf
best_C_log_loss = None
best_C_acc = None
best_C_bal_acc = None

#X_full = adata_train.X  
#y_full = adata_train.obs['identity_layer2']


for C in C_list:
    
    fold_scores_acc = []
    fold_scores_bal_acc = []
    fold_scores_log_loss = []
    
    for train_idx, test_idx in kf.split(adata_train.X):
        adata_train_fold = adata_train[train_idx].copy()
        adata_test_fold = adata_train[test_idx].copy()

    #for train_index, test_index in kf.split(X_full):
    #    X_train, X_test = X_full[train_index], X_full[test_index]
    #    y_train, y_test = y_full[train_index], y_full[test_index]

    #    adata_train_fold = AnnData(X_train)
    #    adata_train_fold.obs['identity_layer2'] = y_train
        
    #    adata_test_fold = AnnData(X_test)
    #    adata_test_fold.obs['identity_layer2'] = y_test

        start = time.time()
        my_model = celltypist.train(adata_train_fold, key,alpha=1/(C*len(adata_train_fold)), feature_selection=False, check_expression = True,use_SGD = True,mini_batch = True, batch_size=batch_size, batch_number=100)

        end=time.time()
        logger.info("seconds used for training")
        logger.info(end-start)

        logger.info("done creating the my_model")

        start = time.time()
        predictions = celltypist.annotate(adata_test_fold, model = my_model,majority_voting = False,mode = 'best match')
        end = time.time()
        taken = end-start
        logger.info("done predictions with seconds")
        logger.info(taken)


        predicted_labels = predictions.predicted_labels.values  # Array of predicted labels from AnnotationResult
        probability_matrix = predictions.probability_matrix.values  # Probability matrix
        true_labels = adata_test_fold.obs[key]  # Assuming true labels are stored here

        class_names = predictions.probability_matrix.columns  # Get class names from probability matrix columns
        true_labels_numeric = np.array([np.where(class_names == label)[0][0] for label in true_labels])

        log_loss_value = custom_ovr_log_loss(true_labels_numeric, probability_matrix)
        logger.info(f"Custom One-vs-Rest Log Loss: {log_loss_value}")
        fold_scores_log_loss.append(log_loss_value)


        accuracy = accuracy_score(true_labels, predicted_labels)
        logger.info(f"Accuracy: {accuracy}")
        fold_scores_acc.append(accuracy)

        balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
        logger.info(f"balabced acc on test data: {balanced_acc}")
        fold_scores_bal_acc.append(balanced_acc)
        
        macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
        logger.info(f"macro_f1 score on test data: {macro_f1}")
# Generate the classification report as a dictionary
        report = classification_report(true_labels, predicted_labels, output_dict=True)
# Convert the dictionary to a pandas DataFrame
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


        file_name = f'/storage/home/dvl5760/work/our_log_reg/change_coreset/add_loss/test_train_split/reports/classification_report_micedvc_celltypist_01.csv'
# Save the DataFrame as a CSV file
        #report_df.to_csv(file_name, index=True)
        logger.info("done saving the classification report")


#now plotting the confusion matrix
#y axis: true label. x axis: predicted
        classes = np.unique(true_labels)
        cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
                cm,
                annot=True,
                cmap="Blues",
                fmt="d",
                xticklabels=classes,
                yticklabels=classes
            )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix")
        plt.tight_layout()
        cm_file = (
                f"/storage/home/dvl5760/work/our_log_reg/change_coreset/add_loss/"
                f"test_train_split/pdf/confusion_matrix_micedvc_01.pdf"
            )
        #plt.savefig(cm_file)
        plt.close()
        logger.info(f"Confusion matrix saved: {cm_file}")



        logger.info("then lets find the log loss on train data")

        predictions = celltypist.annotate(adata_train_fold, model = my_model,majority_voting = False,mode = 'best match')
        predicted_labels = predictions.predicted_labels.values  # Array of predicted labels from AnnotationResult
        probability_matrix = predictions.probability_matrix.values  # Probability matrix
        true_labels = adata_train_fold.obs[key]  # Assuming true labels are stored here

        class_names = predictions.probability_matrix.columns  # Get class names from probability matrix columns
        true_labels_numeric = np.array([np.where(class_names == label)[0][0] for label in true_labels])

        log_loss_value = custom_ovr_log_loss(true_labels_numeric, probability_matrix)
        logger.info(f"Custom One-vs-Rest Log Loss for train: {log_loss_value}")


    mean_score_log_loss = np.mean(fold_scores_log_loss)
    mean_score_acc = np.mean(fold_scores_acc)
    mean_score_bal_acc = np.mean(fold_scores_bal_acc)
    if mean_score_acc > best_score_acc:
        best_score_acc = mean_score_acc
        best_C_acc = C

    if mean_score_bal_acc > best_score_bal_acc:
        best_score_bal_acc = mean_score_bal_acc
        best_C_bal_acc = C

    if mean_score_log_loss < best_score_log_loss:
        best_score_log_loss = mean_score_log_loss
        best_C_log_loss = C

logger.info(f"Best C log loss: {best_C_log_loss} with score: {best_score_log_loss}")

#logger.info(f"Best C acc: {best_C_acc} with score: {best_score_acc}")
logger.info(f"Best C bal acc: {best_C_bal_acc} with score: {best_score_bal_acc}")

end_cv = time.time()
taken = end_cv-start_cv
logger.info("done cv with seconds")
logger.info(taken)

logger.info("all done")

