#!/usr/bin/env python
import os, json, argparse, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from groupyr import LogisticSGL

RANDOM_STATE = 42

def load_df(path, target):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = load_breast_cancer(as_frame=True).frame  # target = 'target'
        target = "target"
    if target not in df.columns:
        raise SystemExit(f"Target '{target}' non trovato. Colonne: {list(df.columns)}")
    return df, target

def make_groups(feature_names):
    """
    Restituisce:
      - groups: lista di array di indici (formato richiesto da groupyr)
      - mapping: dict nome_gruppo -> id (solo per debug/etichette)
    Regole di raggruppamento per Breast Cancer:
      'mean *'  -> gruppo 'mean'
      '* error' -> gruppo 'error'
      'worst *' -> gruppo 'worst'
      altrimenti -> 'other'
    """
    labels = []
    for f in feature_names:
        if f.startswith("mean "):
            g = "mean"
        elif f.endswith(" error"):
            g = "error"
        elif f.startswith("worst "):
            g = "worst"
        else:
            g = "other"
        labels.append(g)

    labels = np.array(labels)
    unique = sorted(np.unique(labels))
    groups = [np.where(labels == lab)[0] for lab in unique]  # <— LISTA DI INDICI
    mapping = {lab: i for i, lab in enumerate(unique)}
    return groups, mapping


def cv_tune_sgl(X, y, groups, alphas, lambdas, n_splits=5):
    """
    Tuning grid molto semplice su (alpha, lambda) massimizzando ROC-AUC.
    alpha: trade-off L1 vs L2 di gruppo (0=solo group, 1=solo lasso)
    lambda: forza totale della penalizzazione
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    rows = []
    best = {"auc_mean": -np.inf}
    for a in alphas:
        for lam in lambdas:
            aucs = []
            for tr, te in cv.split(X, y):
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticSGL(groups=groups, l1_ratio=a, alpha=lam,
                                        max_iter=5000))
                ])
                pipe.fit(X.iloc[tr], y.iloc[tr])
                s = pipe.predict_proba(X.iloc[te])[:,1]
                aucs.append(roc_auc_score(y.iloc[te], s))
            mu, sd = float(np.mean(aucs)), float(np.std(aucs))
            rows.append({"alpha": a, "lambda": lam, "auc_mean": mu, "auc_std": sd})
            if mu > best["auc_mean"]:
                best = {"alpha": a, "lambda": lam, "auc_mean": mu, "auc_std": sd}
    return best, pd.DataFrame(rows).sort_values(["alpha","lambda"])

def eval_cv(pipe, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    rows = []
    for k,(tr,te) in enumerate(cv.split(X,y), start=1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba = pipe.predict_proba(X.iloc[te])[:,1]
        pred  = (proba >= 0.5).astype(int)
        rows.append({
            "fold": k,
            "auc": float(roc_auc_score(y.iloc[te], proba)),
            "acc": float( (pred==y.iloc[te]).mean() ),
            "f1":  float( f1_score(y.iloc[te], pred) ),
        })
    df = pd.DataFrame(rows)
    return df, {
        "auc_mean": float(df.auc.mean()), "auc_std": float(df.auc.std()),
        "acc_mean": float(df.acc.mean()), "acc_std": float(df.acc.std()),
        "f1_mean":  float(df.f1.mean()),  "f1_std":  float(df.f1.std()),
    }

def stability_selection(X, y, feat_names, groups, alpha, lam, repeats=100, subsample=0.8):
    rng = np.random.RandomState(RANDOM_STATE)
    counts = pd.Series(0.0, index=feat_names)
    n = len(y); m = int(round(n*subsample))
    for _ in range(repeats):
        idx = rng.choice(n, size=m, replace=True)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticSGL(groups=groups, l1_ratio=alpha, alpha=lam,
                                max_iter=5000))
        ])
        pipe.fit(X.iloc[idx], y.iloc[idx])
        # coefficienti dopo lo standard scaler
        coefs = pipe.named_steps["clf"].coef_.ravel()
        nonzero = np.abs(coefs) > 1e-8
        counts.iloc[nonzero] += 1
    stab = (counts/repeats).sort_values(ascending=False).reset_index()
    stab.columns = ["feature","freq"]
    return stab

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--target", default="target")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=100)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--bar_top", type=int, default=15)
    ap.add_argument("--alpha_grid", nargs="+", type=float, default=[0.2,0.4,0.6,0.8])
    ap.add_argument("--lambda_grid", nargs="+", type=float, default=[0.001,0.005,0.01,0.02,0.05,0.1])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df, target = load_df(args.data, args.target)
    y = df[target]
    X = df.drop(columns=[target])
    feat_names = X.columns.tolist()

    groups_vec, group_map = make_groups(feat_names)

    # 1) tuning
    best, grid = cv_tune_sgl(X, y, groups_vec, args.alpha_grid, args.lambda_grid, n_splits=args.n_splits)
    grid.to_csv(os.path.join(args.outdir, "sgl_grid.csv"), index=False)

    # heatmap semplice
    piv = grid.pivot(index="alpha", columns="lambda", values="auc_mean")
    plt.figure(figsize=(7,4))
    plt.imshow(piv.values, aspect="auto", origin="lower")
    plt.yticks(range(len(piv.index)), [f"{x:.2f}" for x in piv.index])
    plt.xticks(range(len(piv.columns)), [str(x) for x in piv.columns], rotation=45)
    plt.colorbar(label="CV ROC-AUC (media)")
    plt.xlabel("lambda")
    plt.ylabel("alpha (L1 ratio)")
    plt.title("SGL — tuning grid (AUC)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "sgl_grid_auc.png"), dpi=150)
    plt.close()

    # 2) fit finale + coefficienti
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticSGL(groups=groups_vec, l1_ratio=best["alpha"], alpha=best["lambda"],
                            max_iter=5000))
    ])
    final.fit(X, y)
    coefs = final.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs, "abs_coef": np.abs(coefs)})
    nonzero = coef_df.query("abs_coef > 1e-8").sort_values("abs_coef", ascending=False)
    coef_df.to_csv(os.path.join(args.outdir, "sgl_coefficients.csv"), index=False)

    # bar plot top-N
    topN = min(args.bar_top, len(nonzero)) if len(nonzero) else 0
    if topN>0:
        top = nonzero.head(topN).iloc[::-1]
        plt.figure(figsize=(8, max(4, 0.4*topN)))
        plt.barh(top["feature"], top["coef"])
        plt.xlabel("Coefficiente (SGL, dati standardizzati)")
        plt.title(f"Top-{topN} feature per |coeff| (SGL)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"sgl_coeffs_top{topN}.png"), dpi=150)
        plt.close()

    # 3) CV metrics
    folds, summ = eval_cv(final, X, y, n_splits=args.n_splits)
    folds.to_csv(os.path.join(args.outdir, "sgl_metrics_folds.csv"), index=False)

    # 4) Stabilità
    stab = stability_selection(X, y, feat_names, groups_vec,
                               alpha=best["alpha"], lam=best["lambda"],
                               repeats=args.n_repeats, subsample=args.subsample)
    stab.to_csv(os.path.join(args.outdir, "sgl_features_stability.csv"), index=False)

    sTop = min(20, len(stab))
    st = stab.head(sTop).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.4*sTop)))
    plt.barh(st["feature"], st["freq"])
    plt.xlabel("Selection frequency (bootstrap)")
    plt.title(f"SGL Stability — Top-{sTop} • alpha={best['alpha']} lambda={best['lambda']}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"sgl_stability_top{sTop}.png"), dpi=150)
    plt.close()

    # 5) riepilogo
    out = {
        "best_alpha": float(best["alpha"]),
        "best_lambda": float(best["lambda"]),
        "cv_auc_mean": summ["auc_mean"], "cv_auc_std": summ["auc_std"],
        "cv_acc_mean": summ["acc_mean"], "cv_acc_std": summ["acc_std"],
        "cv_f1_mean":  summ["f1_mean"],  "cv_f1_std":  summ["f1_std"],
        "n_selected": int((np.abs(coefs)>1e-8).sum()),
        "selected_top": nonzero["feature"].head(20).tolist()
    }
    with open(os.path.join(args.outdir, "sgl_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] best: alpha={best['alpha']} | lambda={best['lambda']}")
    print(f"[OK] CV AUC = {summ['auc_mean']:.3f} ± {summ['auc_std']:.3f}")
    print("File salvati in results/:",
          "sgl_grid.csv, sgl_grid_auc.png, sgl_coefficients.csv, "
          f"{'sgl_coeffs_top'+str(topN)+'.png' if topN>0 else '(nessuna feature non-zero)'}, "
          "sgl_metrics_folds.csv, sgl_features_stability.csv, sgl_stability_top20.png, sgl_summary.json")
if __name__ == "__main__":
    main()

