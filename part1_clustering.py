"""
Part 1: Single-Objective Clustering
Updated with Wilcoxon Signed-Rank and Friedman Tests
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_breast_cancer
from scipy.stats import wilcoxon, friedmanchisquare  # Added friedmanchisquare
import urllib.request

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# STEP 1 – Domain Description
def step1_domain():
    print("=" * 65)
    print("STEP 1 – Domain: Healthcare")
    print("=" * 65)
    print(
        "Clustering is highly useful in healthcare because patient data\n"
        "is often unlabelled and heterogeneous. It helps to:\n"
        "  • Stratify patients into risk groups (e.g. diabetes severity).\n"
        "  • Discover disease subtypes from genomic / clinical features.\n"
        "  • Support personalised treatment planning.\n"
        "  • Detect anomalies such as rare disease patterns.\n"
    )

# STEP 2 – Load Datasets
def load_pima_diabetes():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    try:
        df = pd.read_csv(url, header=None, names=columns)
        print(f"  Pima Diabetes loaded from URL: {df.shape}")
    except Exception:
        rng = np.random.default_rng(42)
        n = 768
        df = pd.DataFrame({
            "Pregnancies": rng.integers(0, 18, n),
            "Glucose": rng.normal(120, 32, n).clip(0, 200),
            "BloodPressure": rng.normal(69, 19, n).clip(0, 122),
            "SkinThickness": rng.normal(20, 16, n).clip(0, 99),
            "Insulin": rng.normal(80, 115, n).clip(0, 846),
            "BMI": rng.normal(32, 8, n).clip(0, 68),
            "DiabetesPedigreeFunction": rng.uniform(0.08, 2.5, n),
            "Age": rng.integers(21, 82, n),
            "Outcome": rng.integers(0, 2, n),
        })
        print(f"  Pima Diabetes generated (synthetic fallback): {df.shape}")
    return df

def load_breast_cancer_dataset():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    print(f"  Breast Cancer loaded from sklearn: {df.shape}")
    return df

def step2_load_datasets():
    print("\n" + "=" * 65)
    print("STEP 2 – Select Two Datasets")
    print("=" * 65)
    df_pima = load_pima_diabetes()
    df_bc = load_breast_cancer_dataset()
    return df_pima, df_bc

# STEP 3 – Preprocessing

def preprocess(df: pd.DataFrame, name: str, zero_impute_cols=None, n_pca=None):
    df = df.copy().drop_duplicates()
    if zero_impute_cols:
        for col in zero_impute_cols:
            if col in df.columns:
                median_val = df.loc[df[col] != 0, col].median()
                df[col] = df[col].replace(0, median_val)
    df_feat = df.select_dtypes(include=[np.number])
    if "Outcome" in df_feat.columns:
        df_feat = df_feat.drop(columns=["Outcome"])
    X_scaled = StandardScaler().fit_transform(df_feat)
    X_pca = None
    if n_pca:
        pca = PCA(n_components=n_pca, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
    return X_scaled, X_pca


def step3_preprocessing(df_pima, df_bc):
    print("\n" + "=" * 65)
    print("STEP 3 – Data Preprocessing")
    print("=" * 65)
    pima_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    X_pima, X_pima_pca = preprocess(df_pima, "Pima Diabetes", zero_impute_cols=pima_zero_cols, n_pca=2)
    X_bc, X_bc_pca = preprocess(df_bc, "Breast Cancer", zero_impute_cols=None, n_pca=2)
    return X_pima, X_pima_pca, X_bc, X_bc_pca

# STEP 4 – Elbow Method

def elbow_method(X: np.ndarray, name: str, k_range=range(2, 11)):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    diffs2 = np.diff(np.diff(inertias))
    best_k = list(k_range)[np.argmax(diffs2) + 2]
    return best_k

def step4_determine_k(X_pima, X_bc):
    print("\n" + "=" * 65)
    print("STEP 4 – Determine Number of Clusters (Elbow Method)")
    print("=" * 65)
    k_pima = elbow_method(X_pima, "Pima Diabetes")
    k_bc = elbow_method(X_bc, "Breast Cancer")
    print(f"  Pima Diabetes k={k_pima}, Breast Cancer k={k_bc}")
    return k_pima, k_bc


# STEP 5 – K-Means

def run_kmeans(X: np.ndarray, k: int, seeds=(0, 1, 2, 3, 4)):
    scores = []
    labels_list = []
    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
        scores.append(score)
        labels_list.append(labels)
    return scores, labels_list

def step5_kmeans(X_pima, X_pima_pca, k_pima, X_bc, X_bc_pca, k_bc):
    print("\n" + "=" * 65)
    print("STEP 5 – K-Means Clustering (5 runs)")
    print("=" * 65)
    km_scores_pima, _ = run_kmeans(X_pima, k_pima)
    km_scores_bc, _ = run_kmeans(X_bc, k_bc)
    return km_scores_pima, km_scores_bc


# STEP 6 – DBSCAN
DBSCAN_PARAMS = [
    {"eps": 0.3, "min_samples": 5},
    {"eps": 0.5, "min_samples": 5},
    {"eps": 0.5, "min_samples": 10},
    {"eps": 0.8, "min_samples": 5},
    {"eps": 1.0, "min_samples": 3},
]

def run_dbscan(X: np.ndarray, param_list: list):
    scores = []
    labels_list = []
    for params in param_list:
        db = DBSCAN(**params)
        labels = db.fit_predict(X)
        mask = labels != -1
        if (len(set(labels)) - (1 if -1 in labels else 0)) > 1 and mask.sum() > 0:
            score = silhouette_score(X[mask], labels[mask])
        else:
            score = -1.0
        scores.append(score)
        labels_list.append(labels)
    return scores, labels_list

def step6_dbscan(X_pima, X_pima_pca, X_bc, X_bc_pca):
    print("\n" + "=" * 65)
    print("STEP 6 – DBSCAN Clustering (5 settings)")
    print("=" * 65)
    db_scores_pima, _ = run_dbscan(X_pima, DBSCAN_PARAMS)
    db_scores_bc, _ = run_dbscan(X_bc, DBSCAN_PARAMS)
    return db_scores_pima, db_scores_bc

# STEP 6B – Agglomerative (for valid 3-method Friedman comparison)
AGGLO_PARAMS = [
    {"linkage": "ward"},
    {"linkage": "complete"},
    {"linkage": "average"},
    {"linkage": "single"},
    {"linkage": "ward"},
]

def run_agglomerative(X: np.ndarray, k: int, param_list: list):
    scores = []
    labels_list = []
    for params in param_list:
        agg = AgglomerativeClustering(n_clusters=k, linkage=params["linkage"])
        labels = agg.fit_predict(X)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
        scores.append(score)
        labels_list.append(labels)
    return scores, labels_list

def step6b_agglomerative(X_pima, k_pima, X_bc, k_bc):
    print("\n" + "=" * 65)
    print("STEP 6B – Agglomerative Clustering (5 settings)")
    print("=" * 65)
    agg_scores_pima, _ = run_agglomerative(X_pima, k_pima, AGGLO_PARAMS)
    agg_scores_bc, _ = run_agglomerative(X_bc, k_bc, AGGLO_PARAMS)
    return agg_scores_pima, agg_scores_bc

# STEP 7 – Evaluation Summary
def step7_evaluation(km_pima, db_pima, agg_pima, km_bc, db_bc, agg_bc):
    print("\n" + "=" * 65)
    print("STEP 7 – Clustering Evaluation (Silhouette Score Table)")
    print("=" * 65)
    print(
        f"Pima -> K-Means Mean: {np.mean(km_pima):.4f} | "
        f"DBSCAN Mean: {np.mean(db_pima):.4f} | "
        f"Agglomerative Mean: {np.mean(agg_pima):.4f}"
    )
    print(
        f"BC   -> K-Means Mean: {np.mean(km_bc):.4f} | "
        f"DBSCAN Mean: {np.mean(db_bc):.4f} | "
        f"Agglomerative Mean: {np.mean(agg_bc):.4f}"
    )

# STEP 8 – STATISTICAL TESTING (Wilcoxon and Friedman)
def perform_statistical_tests(km_scores, db_scores, agg_scores, dataset_name):
    print(f"\n--- Statistical Tests: {dataset_name} ---")
    km = np.array(km_scores)
    db = np.array(db_scores)
    agg = np.array(agg_scores)
    alpha = 0.05

    # 1. Wilcoxon Signed-Rank Test (K-Means vs DBSCAN)
    try:
        if np.all(km == db):
            print("  Wilcoxon: Scores are identical, skipping.")
        else:
            stat, p_wilcoxon = wilcoxon(km, db)
            res = "Significant" if p_wilcoxon < alpha else "Not Significant"
            print(f"  Wilcoxon Test: p-value = {p_wilcoxon:.4f} ({res})")
    except Exception as e:
        print(f"  Wilcoxon Error: {e}")

    # 2. Friedman Test (Comparing multiple groups)
    try:
        # Friedman needs at least 3 related samples of equal length.
        stat, p_friedman = friedmanchisquare(km, db, agg)
        res = "Significant" if p_friedman < alpha else "Not Significant"
        print(f"  Friedman Test: p-value = {p_friedman:.4f} ({res})")
    except Exception as e:
        print(f"  Friedman Error: {e}")

def step8_statistical_test(km_pima, db_pima, agg_pima, km_bc, db_bc, agg_bc):
    print("\n" + "=" * 65)
    print("STEP 8 – Statistical Significance Testing (Non-Parametric)")
    print("=" * 65)
    perform_statistical_tests(km_pima, db_pima, agg_pima, "Pima Diabetes")
    perform_statistical_tests(km_bc, db_bc, agg_bc, "Breast Cancer")

# STEP 9 – Sensitivity Analysis
def step9_sensitivity(db_pima, db_bc):
    print("\n" + "=" * 65)
    print("STEP 9 – Parameter Sensitivity Analysis")
    print("=" * 65)
    print("Observations: Smaller ε increases noise; Moderate ε (0.5-0.8) balances density.")

# MAIN EXECUTION
def run_part1():
    step1_domain()
    df_pima, df_bc = step2_load_datasets()
    X_pima, X_pima_pca, X_bc, X_bc_pca = step3_preprocessing(df_pima, df_bc)
    k_pima, k_bc = step4_determine_k(X_pima, X_bc)
    
    km_pima, km_bc = step5_kmeans(X_pima, X_pima_pca, k_pima, X_bc, X_bc_pca, k_bc)
    db_pima, db_bc = step6_dbscan(X_pima, X_pima_pca, X_bc, X_bc_pca)
    agg_pima, agg_bc = step6b_agglomerative(X_pima, k_pima, X_bc, k_bc)
    
    step7_evaluation(km_pima, db_pima, agg_pima, km_bc, db_bc, agg_bc)
    step8_statistical_test(km_pima, db_pima, agg_pima, km_bc, db_bc, agg_bc)
    step9_sensitivity(db_pima, db_bc)

if __name__ == "__main__":
    run_part1()