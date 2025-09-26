#!/usr/bin/env python
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone

RANDOM_STATE = 42

def load_df(path, target):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = load_breast_cancer(as_frame=True).frame  # target == 'target'
        target = "target"
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' non trovato. Colonne: {list(df.columns)}")
    return df, target

def make_rfecv(C=1.0, min_features=5):
    # scaler in-pipeline, LinearSVC con C variabile
    clf = LinearSVC(dual=False, class_weight="balanced", random_state=RANDOM_STATE, max_iter=10000, C=C)
    rfecv = RFECV(
        estimator=clf,
        step=0.1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        min_features_to_select=min_features,
        n_jobs=-1
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("fs", rfecv)])
    return pipe

def nested_cv(X, y, feat_names, C_grid=(0.01, 0.1, 1.0, 10.0, 100.0), min_features=5, n_outer=5):
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=RANDOM_STATE)
    outer_metrics = []
    chosen_C = []
    chosen_k = []
    selected_feats_outer = []

    for outer_fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        # inner search su C: addestra RFECV per ogni C e sceglie quello con migliore AUC media
        best_auc, best_C, best_model = -np.inf, None, None
        best_support = None
        for C in C_grid:
            model = make_rfecv(C=C, min_features=min_features)
            model.fit(Xtr, ytr)
            auc_mean = model.named_steps["fs"].grid_scores_.max() if hasattr(model.named_steps["fs"], "grid_scores_") else \
                       model.named_steps["fs"].cv_results_["mean_test_score"].max()
            if auc_mean > best_auc:
                best_auc, best_C, best_model = auc_mean, C, model
                best_support = model.named_steps["fs"].support_

        # valuta sul test esterno
        yprob = best_model.predict(Xte)  # LinearSVC non ha predict_proba; usiamo decision_function per AUC
        if hasattr(best_model.named_steps["fs"].estimator_, "decision_function"):
            # rifit per ottenere decision_function coerente (già fit), poi compute su Xte
            # (passiamo dall'intera pipeline)
            dec = best_model.decision_function(Xte)
            auc = roc_auc_score(yte, dec)
        else:
            auc = np.nan

        yhat = best_model.predict(Xte)
        acc = accuracy_score(yte, yhat)
        f1  = f1_score(yte, yhat)

        k = best_model.named_steps["fs"].n_features_
        sel_names = [n for n, s in zip(feat_names, best_support) if s]

        outer_metrics.append({"fold": outer_fold, "C": best_C, "k": int(k), "acc": acc, "f1": f1, "auc": float(auc)})
        chosen_C.append(best_C); chosen_k.append(k)
        selected_feats_outer.append(sel_names)

    metrics_df = pd.DataFrame(outer_metrics)
    return metrics_df, selected_feats_outer, chosen_C, chosen_k

def selection_stability(X, y, feat_names, C, min_features=5, n_repeats=100, subsample=0.8):
    rng = np.random.RandomState(RANDOM_STATE)
    counts = pd.Series(0, index=feat_names, dtype=float)

    n = len(y)
    m = int(round(n * subsample))
    for r in range(n_repeats):
        idx = rng.choice(n, size=m, replace=True)
        Xb, yb = X.iloc[idx], y.iloc[idx]
        model = make_rfecv(C=C, min_features=min_features)
        model.fit(Xb, yb)
        support = model.named_steps["fs"].support_
        counts.iloc[support] += 1

    freq = counts / n_repeats
    stab_df = freq.sort_values(ascending=False).reset_index()
    stab_df.columns = ["feature", "freq"]
    return stab_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--target", default="target")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--min_features", type=int, default=5)
    ap.add_argument("--n_outer", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=100)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--bar_top", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df, target = load_df(args.data, args.target)
    y = df[target]
    X = df.drop(columns=[target])
    feat_names = X.columns.tolist()

    # 1) nested CV (outer perf + inner tuning di C e k)
    metrics_df, selected_feats_outer, chosen_C, chosen_k = nested_cv(
        X, y, feat_names, C_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        min_features=args.min_features, n_outer=args.n_outer
    )
    metrics_csv = os.path.join(args.outdir, "metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    summary = metrics_df[["acc","f1","auc"]].agg(["mean","std"]).round(4)
    print("\n[Outer CV] performance summary:\n", summary)

    # 2) stabilità: usa la mediana dei C scelti come C “tipico”
    typical_C = float(np.median(chosen_C))
    stab_df = selection_stability(
        X, y, feat_names, C=typical_C,
        min_features=args.min_features, n_repeats=args.n_repeats, subsample=args.subsample
    )
    stab_csv = os.path.join(args.outdir, "features_selected.csv")
    stab_df.to_csv(stab_csv, index=False)

    # 3) bar plot top-N per stabilità
    topN = min(args.bar_top, len(stab_df))
    top = stab_df.head(topN).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.4*topN)))
    plt.barh(top["feature"], top["freq"])
    plt.xlabel("Selection frequency (bootstrap)")
    plt.title(f"Top-{topN} stable features • C≈{typical_C}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"stability_top{topN}.png"), dpi=150)
    plt.close()

    # 4) salva anche un piccolo JSON di riepilogo
    out = {
        "outer_cv": {
            "acc_mean": float(metrics_df["acc"].mean()),
            "acc_std": float(metrics_df["acc"].std()),
            "f1_mean":  float(metrics_df["f1"].mean()),
            "f1_std":  float(metrics_df["f1"].std()),
            "auc_mean": float(metrics_df["auc"].mean()),
            "auc_std":  float(metrics_df["auc"].std())
        },
        "chosen_C_median": typical_C,
        "chosen_k_median": int(np.median(chosen_k))
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n[OK] Salvati:")
    print(" -", metrics_csv)
    print(" -", stab_csv)
    print(" -", os.path.join(args.outdir, f"stability_top{topN}.png"))
    print(" -", os.path.join(args.outdir, "summary.json"))

if __name__ == "__main__":
    main()

