#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import random

# Constants
M_VALUES = {'HIV-protease': 8, 'T-cell epitopes': 9, 'HLA binding': 9}
NUM_PEPTIDES = {'HIV-protease': 746, 'T-cell epitopes': 927, 'HLA binding': 3000}
NUM_PROPERTIES_RANGE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50]
CLASSIFICATION_PROBLEMS = ['HIV-protease', 'T-cell epitopes', 'HLA binding']
CROSS_VALIDATION_FOLDS = 10

# Load datasets
peptide_data = pd.read_csv('datasets/synthetic/dataset1_peptides.csv')

# Placeholder for physicochemical properties (should be loaded or defined here)
# Example: properties = pd.read_csv('path_to_properties.csv')
properties = np.random.rand(50, 20)  # 50 properties, 20 amino acids

# Map amino acids to indices (standard 20 amino acids)
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

# Function to compute Pp_20+k
def compute_pp_k(M, k, properties, sequences):
    results = []
    for seq in sequences:
        pp_k_values = []
        # Iterate over properties (rows)
        for p in range(properties.shape[0]):
            sum_diff = 0
            for a in range(M - k):
                # Map amino acid characters to indices
                aa1_idx = aa_to_idx.get(seq[a], 0)
                aa2_idx = aa_to_idx.get(seq[a + k], 0)
                sum_diff += properties[p, aa1_idx] - properties[p, aa2_idx]
            pp_k_values.append(sum_diff / (M - k))
        results.append(pp_k_values)
    return np.array(results)

# Main simulation function
def main():
    for problem in CLASSIFICATION_PROBLEMS:
        M = M_VALUES[problem]
        num_peptides = NUM_PEPTIDES[problem]
        # Filter sequences by length M
        filtered_data = peptide_data[peptide_data['sequence'].str.len() == M].copy()
        if len(filtered_data) == 0:
            print(f'Warning: No sequences of length {M} found for {problem}')
            continue
        
        for num_properties in NUM_PROPERTIES_RANGE:
            auc_scores = []
            for _ in range(10):  # Repeat with different random selections of properties
                # Select rows (properties), not columns (amino acids)
                selected_property_indices = np.random.choice(properties.shape[0], num_properties, replace=False)
                selected_properties = properties[selected_property_indices, :]
                feature_matrix = []
                for k in range(1, M):
                    encoded_features = compute_pp_k(M, k, selected_properties, filtered_data['sequence'])
                    feature_matrix.append(encoded_features)
                feature_matrix = np.concatenate(feature_matrix, axis=1)

                # Cross-validation
                skf = StratifiedKFold(n_splits=CROSS_VALIDATION_FOLDS)
                fold_auc = []
                for train_index, test_index in skf.split(feature_matrix, filtered_data['label']):
                    X_train, X_test = feature_matrix[train_index], feature_matrix[test_index]
                    y_train, y_test = filtered_data['label'].iloc[train_index], filtered_data['label'].iloc[test_index]
                    clf = SVC(kernel='rbf', probability=True)
                    clf.fit(X_train, y_train)
                    y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Get probability of positive class
                    fold_auc.append(roc_auc_score(y_test, y_pred_proba))
                auc_scores.append(np.mean(fold_auc))
            print(f'Classification problem: {problem}, Num properties: {num_properties}, Mean AUC: {np.mean(auc_scores):.4f}, Std: {np.std(auc_scores):.4f}')

if __name__ == '__main__':
    main()