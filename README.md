# SVM-RFE (Breast Cancer) — Demo
Selezione di feature con **SVM-RFE (LinearSVC)** su dataset pubblico.  
Pipeline: `StandardScaler → RFECV(LinearSVC)` • Metrica: **ROC-AUC (5-fold)**.

## Esecuzione
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
python src/rfe_simple.py --min_features 5

## Output
- results/rfe_features.csv
- results/rfe_nfeat_vs_auc.png
- results/metrics_rfe.json
- results/svm_coefficients.csv + rfe_coeffs_topN.png
