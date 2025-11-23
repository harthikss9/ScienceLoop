#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent hanging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import random

# Constants
NUM_PHYSICOCHEMICAL_PROPERTIES = 7  # Example: typically between 5 and 10
MAX_LAG_K = 5  # Example: between 1 and min(5, M-1)
NUM_PEPTIDES_TRAIN = 1500
NUM_PEPTIDES_TEST = 300
POSITIVE_CLASS_RATIO = 0.4
AMINO_ACID_ALPHABET_SIZE = 20

# Amino acids
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

# Physicochemical properties for each amino acid, normalized
properties = {
    'hydrophobicity': np.random.rand(AMINO_ACID_ALPHABET_SIZE),
    'charge': np.random.rand(AMINO_ACID_ALPHABET_SIZE),
    'polarity': np.random.rand(AMINO_ACID_ALPHABET_SIZE),
    'volume': np.random.rand(AMINO_ACID_ALPHABET_SIZE),
    'surface_area': np.random.rand(AMINO_ACID_ALPHABET_SIZE),
    'flexibility': np.random.rand(AMINO_ACID_ALPHABET_SIZE),
    'transfer_free_energy': np.random.rand(AMINO_ACID_ALPHABET_SIZE)
}

# Function to generate random peptides
def generate_peptides(num_peptides, length):
    return [''.join(random.choices(amino_acids, k=length)) for _ in range(num_peptides)]

# Function to generate labels
def generate_labels(num_peptides, positive_ratio):
    num_positives = int(num_peptides * positive_ratio)
    return np.array([1] * num_positives + [0] * (num_peptides - num_positives))

# Autocorrelation encoding
def autocorrelation_encoding(peptides, num_properties, max_lag):
    encoded_features = []
    for peptide in peptides:
        feature_vector = []
        M = len(peptide)
        for prop in properties:
            prop_values = properties[prop]
            autocorr_features = []
            for k in range(1, max_lag + 1):
                sum_diff = 0
                for i in range(M - k):
                    val_a = prop_values[amino_acids.index(peptide[i])]
                    val_b = prop_values[amino_acids.index(peptide[i + k])]
                    sum_diff += (val_a - val_b)
                autocorr_features.append(sum_diff / (M - k))
            feature_vector.extend(autocorr_features)
        encoded_features.append(feature_vector)
    return np.array(encoded_features)

# Orthonormal encoding
def orthonormal_encoding(peptides):
    encoder = OneHotEncoder(categories=[list(amino_acids)] * len(peptides[0]))
    return encoder.fit_transform([list(peptide) for peptide in peptides]).toarray()

# Main simulation function
def main():
    peptide_lengths = [8, 9, 10, 11, 12, 15]
    encoding_methods = ['autocorrelation_physicochemical', 'orthonormal']
    classifier_types = ['SVM', 'Random_Forest', 'Logistic_Regression', 'Neural_Network']
    num_properties_used = [3, 5, 7, 10]
    dataset_sizes = [500, 1000, 1500, 2000]

    print("Starting simulation...", flush=True)
    results = []
    total_iterations = len(peptide_lengths) * len(encoding_methods) * len(classifier_types)
    current_iteration = 0
    
    for M in peptide_lengths:
        print(f"Processing peptide length: {M}", flush=True)
        peptides_train = generate_peptides(NUM_PEPTIDES_TRAIN, M)
        peptides_test = generate_peptides(NUM_PEPTIDES_TEST, M)
        labels_train = generate_labels(NUM_PEPTIDES_TRAIN, POSITIVE_CLASS_RATIO)
        labels_test = generate_labels(NUM_PEPTIDES_TEST, POSITIVE_CLASS_RATIO)

        for encoding_method in encoding_methods:
            if encoding_method == 'autocorrelation_physicochemical':
                X_train = autocorrelation_encoding(peptides_train, NUM_PHYSICOCHEMICAL_PROPERTIES, MAX_LAG_K)
                X_test = autocorrelation_encoding(peptides_test, NUM_PHYSICOCHEMICAL_PROPERTIES, MAX_LAG_K)
            else:
                X_train = orthonormal_encoding(peptides_train)
                X_test = orthonormal_encoding(peptides_test)

            for classifier_type in classifier_types:
                current_iteration += 1
                print(f"  [{current_iteration}/{total_iterations}] {encoding_method} - {classifier_type}", flush=True)
                
                if classifier_type == 'SVM':
                    classifier = SVC(probability=True)
                elif classifier_type == 'Random_Forest':
                    classifier = RandomForestClassifier()
                elif classifier_type == 'Logistic_Regression':
                    classifier = LogisticRegression()
                else:  # Neural Network
                    classifier = MLPClassifier(max_iter=200)  # Limit iterations to prevent hanging

                classifier.fit(X_train, labels_train)
                probs = classifier.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(labels_test, probs)
                roc_auc = auc(fpr, tpr)
                euc = 1 - roc_auc
                results.append((M, encoding_method, classifier_type, roc_auc, euc))
    
    print("Simulation complete. Generating plot...", flush=True)

    # Plotting results
    plt.figure(figsize=(10, 8))
    for result in results:
        plt.plot(result[0], result[4], label=f'{result[1]}-{result[2]} AUC: {result[3]:.2f}')
    plt.xlabel('Peptide Length')
    plt.ylabel('EUC')
    plt.title('EUC by Peptide Length and Classifier')
    plt.legend()
    plt.savefig('ml_classification_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to: ml_classification_results.png")
    plt.close()

if __name__ == '__main__':
    main()