#!/usr/bin/env python
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.datasets import load_breast_cancer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="CSV opzionale; se assente usa il dataset breast_cancer di scikit-learn")
    ap.add_argument("--target", default="target")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--min_features", type=int, default=5)
    ap.add_argument("--bar_top", type=int, default=15,
                    help="Numero di feature da mostrare nel bar plot (ordinate per |coef|)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- load
    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
    else:
        # dataset pubblico di scikit-learn
        df = load_breast_cancer(as_frame=True).frame  # la colonna target qui è già 'target'

    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' non trovato. Colonne: {list(df.columns)}")

    y = df[args.target].values
    X = df.drop(columns=[args.target])
    feat_names = X.columns.tolist()

    # --- model + RFECV (SVM lineare + scaler in pipeline, niente leakage)
    clf = LinearSVC(dual=False, class_weight="balanced", random_state=0, max_iter=10000)
    rfecv = RFECV(
        estimator=clf,
        step=0.1,  # rimuove ~10% delle feature per passo
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        scoring="roc_auc",
        min_features_to_select=args.min_features,
        n_jobs=-1
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("fs", rfecv)])
    pipe.fit(X, y)

    fs = pipe.named_steps["fs"]
    support = fs.support_
    ranking = fs.ranking_
    n_best = fs.n_features_
    selected = [f for f, s in zip(feat_names, support) if s]

    # --- salva tabella feature (support/ranking)
    (pd.DataFrame({"feature": feat_names, "selected": support, "ranking": ranking})
       .sort_values(["selected", "ranking"], ascending=[False, True])
       .to_csv(os.path.join(args.outdir, "rfe_features.csv"), index=False))

    # --- curva CV: #feature vs AUC media
    cv_scores = fs.cv_results_["mean_test_score"]
    n_feat_grid = fs.cv_results_["n_features"]

    plt.figure(figsize=(6, 4))
    plt.plot(n_feat_grid, cv_scores, marker="o")
    plt.axvline(n_best, linestyle="--")
    plt.xlabel("# features selezionate")
    plt.ylabel("CV ROC-AUC (media)")
    plt.title("SVM-RFE (RFECV)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "rfe_nfeat_vs_auc.png"), dpi=150)
    plt.close()

    # --- performance outer CV rapida sull'intera pipeline
    auc5 = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    metrics = {
        "n_features_selected": int(n_best),
        "selected_top": selected[:20],
        "cv_auc_mean": float(np.mean(auc5)),
        "cv_auc_std": float(np.std(auc5))
    }
    with open(os.path.join(args.outdir, "metrics_rfe.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- COEFFICIENTI SVM sul sotto-spazio selezionato
    # fs.estimator_ è la SVM lineare addestrata dopo la selezione finale
    # NB: i coefficienti sono nel dominio delle feature scalate (va bene per ranking/interpretazione relativa)
    try:
        coef = fs.estimator_.coef_.ravel()
    except AttributeError as e:
        raise SystemExit("Impossibile accedere ai coefficienti della SVM. "
                         "Assicurati di usare LinearSVC come estimator in RFECV.") from e

    sel_names = np.array(feat_names)[support]
    coef_df = pd.DataFrame({
        "feature": sel_names,
        "coef": coef,
        "abs_coef": np.abs(coef)
    }).sort_values("abs_coef", ascending=False)

    coef_csv = os.path.join(args.outdir, "svm_coefficients.csv")
    coef_df.to_csv(coef_csv, index=False)

    # --- BAR PLOT Top-N coefficienti per |peso|
    topN = min(args.bar_top, len(coef_df))
    top_df = coef_df.head(topN).iloc[::-1]  # inverti per avere il più grande in alto nel barh
    plt.figure(figsize=(8, max(4, 0.4 * topN)))
    plt.barh(top_df["feature"], top_df["coef"])
    plt.xlabel("Coefficiente (LinearSVC)")
    plt.title(f"Top-{topN} feature per |coefficiente| (SVM dopo RFE)")
    plt.tight_layout()
    bar_path = os.path.join(args.outdir, f"rfe_coeffs_top{topN}.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()

    print(f"[OK] Selezionate {n_best} feature")
    print(f"[OK] CV AUC = {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    print(f"[OK] File salvati in '{args.outdir}/':")
    print("     - rfe_features.csv")
    print("     - rfe_nfeat_vs_auc.png")
    print("     - metrics_rfe.json")
    print("     - svm_coefficients.csv")
    print(f"     - {os.path.basename(bar_path)}")

if __name__ == "__main__":
    main()

