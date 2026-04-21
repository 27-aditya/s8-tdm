# Assignment II – Walkthrough
Course: Topics in Data Mining | Total Marks: 20

---

## What Was Built

| File | Purpose |
|---|---|
| `part1_clustering.py` | Full Part 1 pipeline (K-Means + DBSCAN) |
| `part2_multiobjective.py` | Part 2 – NSGA-II from scratch |
| `run_all.py` | Entry point – runs both parts |
| `requirements.txt` | Python dependencies |
| `venv/` | Python virtual environment |

---

## Part 1: Single-Objective Clustering ✅

**Domain**: Healthcare — patient stratification via clustering

**Datasets**:
- **Dataset 1**: Pima Indians Diabetes (UCI) – 768 samples × 8 features
- **Dataset 2**: Breast Cancer Wisconsin (sklearn) – 569 samples × 30 features

**Preprocessing** (Step 3):
- Zero-imputation on biologically impossible zero values (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
- StandardScaler normalization
- PCA (2 components) for visualization

**Elbow Method** (Step 4):
- Plots `elbow_Pima_Diabetes.png` and `elbow_Breast_Cancer.png` showing inertia vs k
- Optimal k selected automatically via max second-derivative heuristic

**K-Means Clustering** (Step 5):
- 5 runs with different random seeds (0–4)
- Silhouette score recorded per run
- Best run visualized in PCA space → `kmeans_pima.png`, `kmeans_bc.png`

**DBSCAN Clustering** (Step 6):
- 5 parameter settings:

| Run | ε   | MinPts |
|-----|-----|--------|
| 1   | 0.3 | 5      |
| 2   | 0.5 | 5      |
| 3   | 0.5 | 10     |
| 4   | 0.8 | 5      |
| 5   | 1.0 | 3      |

- Best run visualized → `dbscan_pima.png`, `dbscan_bc.png`

**Evaluation – Silhouette Score** (Step 7):
- All runs tabulated; comparison plots → `silhouette_comparison_Pima_Diabetes.png`, `silhouette_comparison_Breast_Cancer.png`

**Statistical Test – Wilcoxon Signed-Rank Test** (Step 8):
- H₀: No significant difference between K-Means and DBSCAN silhouette scores
- p-value printed at runtime; α = 0.05

**Parameter Sensitivity Analysis** (Step 9):
- Bar chart of DBSCAN silhouette score vs parameter settings → `dbscan_sensitivity.png`
- Observation: moderate ε (0.5–0.8) with MinPts=5 gives best results; too small ε creates noise, too large merges all points

---

## Part 2: Multi-Objective Clustering (NSGA-II) ✅

**Dataset**: Breast Cancer Wisconsin (same as Part 1)

**Objectives** (both minimized, continuous, normalized, with linear trade-off):

| Objective | Description | Direction |
|---|---|---|
| f₁ | Intra-cluster compactness (avg squared dist to centroid) | Minimize |
| f₂ | Number of clusters k | Minimize |

> Fewer clusters → worse compactness → these objectives conflict ✓

**Chromosome Encoding**: `[k ∈ [2,15], seed ∈ [0,999]]` → K-Means evaluation

**NSGA-II Algorithm** (from scratch):
- Population: 60 individuals, 80 generations
- Fast non-dominated sort (O(MN²))
- Crowding distance for diversity preservation
- Binary tournament selection (k=3)
- Single-point crossover (rate: 0.85) + random mutation (rate: 0.15)

**Pareto Front** → `pareto_front.png`
- X-axis: Number of clusters (f₂)
- Y-axis: Intra-cluster compactness (f₁)
- Non-dominated solutions shown in red; dominated in blue

**Convergence** → `nsga2_convergence.png`

---

## Execution Results

```
Total runtime: 21.2s   |   Exit code: 0
```

**11 plots generated** in `/home/aditya/tdm2/`:

| Plot | Description |
|---|---|
| `elbow_Pima_Diabetes.png` | Elbow curve for Dataset 1 |
| `elbow_Breast_Cancer.png` | Elbow curve for Dataset 2 |
| `kmeans_pima.png` | K-Means clusters (PCA) – Pima |
| `kmeans_bc.png` | K-Means clusters (PCA) – Breast Cancer |
| `dbscan_pima.png` | DBSCAN clusters (PCA) – Pima |
| `dbscan_bc.png` | DBSCAN clusters (PCA) – Breast Cancer |
| `dbscan_sensitivity.png` | Sensitivity of DBSCAN to parameters |
| `silhouette_comparison_Pima_Diabetes.png` | K-Means vs DBSCAN silhouette |
| `silhouette_comparison_Breast_Cancer.png` | K-Means vs DBSCAN silhouette |
| `pareto_front.png` | NSGA-II Pareto front |
| `nsga2_convergence.png` | NSGA-II convergence per generation |

---

## How to Run

```bash
cd /home/aditya/tdm2
./venv/bin/python run_all.py
```
