#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
K_NEIGHBORS = [3, 5, 7, 9, 11]
TRAIN_TEST_SPLIT_RATIO = 0.75
N_SAMPLES = 195
BASELINE_FEATURE_COUNT = 16
RANDOM_SEED = 42

# Load dataset
data = pd.read_csv('datasets/synthetic/classification_data.csv')

# Functions
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def mode(labels):
    return np.bincount(labels).argmax()

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1

def knn_classification(X_train, y_train, X_test, y_test, k):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    return compute_metrics(y_test, y_pred)

def main():
    results = []
    for k in K_NEIGHBORS:
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED)
        acc, prec, rec, f1 = knn_classification(X_train, y_train, X_test, y_test, k)
        results.append((k, acc, prec, rec, f1))
    # Plotting
    ks, accs, precs, recs, f1s = zip(*results)
    plt.figure(figsize=(10, 5))
    plt.plot(ks, accs, label='Accuracy')
    plt.plot(ks, precs, label='Precision')
    plt.plot(ks, recs, label='Recall')
    plt.plot(ks, f1s, label='F1 Score')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Scores')
    plt.title('KNN Performance Metrics')
    plt.legend()
    plt.savefig('knn_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()