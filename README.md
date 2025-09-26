# SVM-RFE (Breast Cancer) — Demo rigorosa

![CI](https://github.com/<TUO_USER>/<TUO_REPO>/actions/workflows/run.yml/badge.svg)

Selezione di feature con **SVM-RFE (LinearSVC)** su dataset pubblico *Breast Cancer Wisconsin (Diagnostic)*.  
Obiettivi:
- Pipeline **senza leakage**: `StandardScaler → RFECV(LinearSVC)`
- **Valutazione** con ROC-AUC 5-fold (demo) + **Nested CV** (outer perf, inner tuning)
- **Interpretabilità**: coefficienti SVM sulle feature selezionate (bar plot)
- **Stabilità**: frequenza di selezione via bootstrap/subsampling

## Dataset
- Se non fornisci un CSV, lo script usa `sklearn.datasets.load_breast_cancer()` (target = `target`).
- Se usi un tuo CSV, specifica `--data` e `--target`.

## Metodo
### Demo RFE
- RFECV (rimozione ~10% per passo), scoring = ROC-AUC.
- Output: curva `#features vs AUC`, lista feature selezionate, coeff SVM e relativo bar plot.

### Nested CV + Stabilità
- **Outer CV (5-fold)**: stima onesta di ACC/F1/AUC.
- **Inner CV**: sceglie automaticamente **C** della SVM e il numero di feature **k**.
- **Bootstrap (100× @ 0.8)**: stima **selection frequency** per ogni feature.

## Requisiti (consigliati)
Python 3.11/3.12
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

