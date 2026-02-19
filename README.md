# Census Income Classification & Segmentation

This project delivers two models for a retail marketing use case using weighted U.S. Census data:

1. **Classification:** Predict whether a person’s income is **≤ $50K** or **> $50K** using 40 demographic and employment variables.
2. **Segmentation:** Cluster the population into segments and describe how they differ for marketing.

---

## Repository layout (Census_Data)

- **`census-bureau.data`** - Comma-delimited data (no header). Each row has 40 features, a weight, and a label.
- **`census-bureau.columns`** - Column names (one per line), in the same order as the data file.
- **`Census_Data _Classification.ipynb`** - Classification model: training, evaluation, and comparison (Logistic Regression, Random Forest, XGBoost).
- **`Census_Data_Segmentation.ipynb`** - Segmentation model: KMeans clustering, segment profiles, PCA visualization, and optional 3-PC clustering view.
- **`Business_Report_CensusData.pdf`** - Business report summarizing classification and segmentation results for the client.
- **`README.md`** - This file.
- **`requirements.txt`** - Python dependencies.

Plots are written to a **`plots/`** directory created by the notebooks.

---

## Requirements

- **Python:** 3.8+
- **Libraries:**  
  `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `scipy`

Install from this directory:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost scipy
```

---

## How to run

1. **Put the data files in this (Census_Data) directory**  
   Copy `census-bureau.data` and `census-bureau.columns` into the Census_Data folder. The notebooks use `DATA_DIR = Path("./")`, so they expect these files in the current working directory.

2. **Start Jupyter from Census_Data**  
   ```bash
   cd /path/to/Census_Data
   jupyter notebook
   ```
   Or open the Census_Data folder in JupyterLab / VS Code and run from there.

3. **Run the notebooks**  
   - **Classification:** Open `Census_Data _Classification.ipynb` and run all cells (*Run All*).  
   - **Segmentation:** Open `Census_Data_Segmentation.ipynb` and run all cells.

---

## What each notebook does

### Classification notebook

- Loads data using column names from `census-bureau.columns`.
- Builds binary target: income > $50K (1) vs ≤ $50K (0).
- Preprocesses numeric and categorical features (scaling, one-hot encoding).
- Uses census **weight** for sample weighting in training and for weighted ROC-AUC and PR-AUC.
- Trains and compares: Logistic Regression (with class weighting), Random Forest (with optional threshold tuning), XGBoost (with `scale_pos_weight`).
- Reports classification metrics, confusion matrices, and weighted ROC-AUC and PR-AUC.

### Segmentation notebook

- Loads the same data and builds a numeric feature set (e.g. age, weeks worked, wage, capital gains/losses, dividends, etc.).
- Preprocesses (log-transform, winsorize, standardize) and runs **KMeans** on the full feature space (no sample weights in KMeans to avoid numerical issues).
- Selects number of clusters using elbow and silhouette plots; uses **weight** only for segment-size and post-hoc analysis.
- Assigns segments, profiles them (e.g. mean features and income mix per cluster), and visualizes with PCA.
- Includes hierarchical clustering (dendrogram) on a sample for validation.

---

## Outputs

- **Classification:** Trained models and evaluation printed in the notebook; figures saved under `plots/`.
- **Segmentation:** Cluster labels, segment profiles, and plots (elbow, silhouette, PCA) in the notebook and under `plots/`.

---

## Data note

The **weight** column reflects the relative number of people in the population that each record represents. It is used for:
- **Classification:** sample weights in training and in weighted evaluation metrics.
- **Segmentation:** post-clustering analysis (e.g. weighted segment sizes and income mix), not inside KMeans.

---

## References

1. Mitchell, T. (1997). Machine Learning. McGraw-Hill.
2. Scikit-learn Machine Learning Library Documentation
3. Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.). Springer.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD. XGBoost Documentation

