import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare

def perform_wilcoxon_test(km_scores, db_scores, dataset_name):
    """
    Wilcoxon Signed-Rank Test: Compares two related samples (K-Means vs DBSCAN).
    """
    print(f"\n  --- Wilcoxon Signed-Rank Test: {dataset_name} ---")
    km = np.array(km_scores)
    db = np.array(db_scores)
    
    # Check if there are differences to test
    if np.all(km == db):
        print("    All scores are identical; skipping Wilcoxon test.")
        return

    stat, p_value = wilcoxon(km, db, alternative='two-sided')
    alpha = 0.05
    
    print(f"    Statistic: {stat:.4f}")
    print(f"    p-value:   {p_value:.4f}")
    
    if p_value < alpha:
        print(f"    Result: REJECT H0 - Significant difference found (p < {alpha}).")
    else:
        print(f"    Result: FAIL TO REJECT H0 - No significant difference found.")

def perform_friedman_test(km_scores, db_scores, dataset_name):
    """
    Friedman Test: Compares K-Means and all DBSCAN parameter settings.
    """
    print(f"\n  --- Friedman Test: {dataset_name} ---")
    
    # Each list represents a 'treatment'. We compare K-Means results against DBSCAN results.
    try:
        stat, p_value = friedmanchisquare(km_scores, db_scores)
        alpha = 0.05
        
        print(f"    Statistic: {stat:.4f}")
        print(f"    p-value:   {p_value:.4f}")
        
        if p_value < alpha:
            print(f"    Result: REJECT H0 - Significant difference across configurations.")
        else:
            print(f"    Result: FAIL TO REJECT H0 - No significant difference across configurations.")
    except ValueError as e:
        print(f"    Friedman test error: {e}")