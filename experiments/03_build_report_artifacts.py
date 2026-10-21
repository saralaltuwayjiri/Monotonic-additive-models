"""
One consolidated, deterministic run (fixed seed=42) that produces every
number and figure used in the manuscript's Results section. This is the
canonical script: its output is what the manuscript now reports, and it
is the exact code shipped alongside the paper.

Usage: python3 experiments/03_build_report_artifacts.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from src.preprocess import load_and_preprocess
from src.models import (
    tune_hyperparameters, build_models, repeated_cv_scores,
    paired_tests, RANDOM_SEED,
)
from src.shap_lime_impl import ShapLimeExplainer


def top_k_jaccard(attr_a, attr_b, feature_names, k=5):
    top_a = set(np.array(feature_names)[np.argsort(-np.abs(attr_a))[:k]])
    top_b = set(np.array(feature_names)[np.argsort(-np.abs(attr_b))[:k]])
    return len(top_a & top_b) / len(top_a | top_b)


def main():
    os.makedirs("outputs", exist_ok=True)
    X, y, feature_names = load_and_preprocess("data/student-mat.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    tuned = tune_hyperparameters(X_train, y_train)
    models = build_models(tuned)

    # ---- Table 2: hyperparameter search ----
    table2 = {
        name: {"best_params": info["best_params"],
               "best_cv_roc_auc": info["best_cv_score"]}
        for name, info in tuned.items()
    }

    # ---- Table 3: repeated CV + significance ----
    scores = repeated_cv_scores(models, X, y, tuned["lr"]["scaler"])
    table3 = {name: {"mean": float(s.mean()), "std": float(s.std())}
              for name, s in scores.items()}
    sig = {
        "rf_vs_gb": paired_tests(scores["rf"], scores["gb"]),
        "rf_vs_lr": paired_tests(scores["rf"], scores["lr"]),
        "lr_vs_gb": paired_tests(scores["lr"], scores["gb"]),
    }

    # ---- Section 4.4: illustrative single split + Figure 2 ----
    scaler = tuned["lr"]["scaler"]
    illus = {}
    cms = {}
    fitted = {}
    for name, model in models.items():
        Xtr = scaler.transform(X_train) if name == "lr" else X_train.values
        Xte = scaler.transform(X_test) if name == "lr" else X_test.values
        model.fit(Xtr, y_train)
        fitted[name] = model
        proba = model.predict_proba(Xte)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, proba)
        preds = (proba >= 0.5).astype(int)
        cm = sk_confusion_matrix(y_test, preds)
        illus[name] = {"test_auc": float(auc)}
        cms[name] = cm

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = {"lr": "Logistic Regression", "rf": "Random Forest", "gb": "Gradient Boosting"}
    for ax, name in zip(axes, ["lr", "rf", "gb"]):
        cm = cms[name]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{labels[name]}\nAUC={illus[name]['test_auc']:.3f}")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Fail", "Pass"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Fail", "Pass"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig("outputs/figure2_confusion_matrices.png", dpi=300)
    plt.close()

    # ---- Section 4.5/4.6/4.7: explainability (use tuned RF fit on train) ----
    rf = fitted["rf"]
    predict_fn = lambda Z: rf.predict_proba(Z)[:, 1]
    explainer = ShapLimeExplainer(predict_fn, X_train.values,
                                   n_features=X.shape[1], seed=RANDOM_SEED,
                                   background_size=50)

    X_test_arr = X_test.values
    y_test_arr = y_test.values
    proba_test = predict_fn(X_test_arr)

    n_global = min(50, len(X_test_arr))
    rng = np.random.RandomState(RANDOM_SEED)
    global_idx = rng.choice(len(X_test_arr), n_global, replace=False)

    shap_matrix = np.array([explainer.kernel_shap(X_test_arr[i]) for i in global_idx])
    global_shap_importance = np.abs(shap_matrix).mean(axis=0)
    global_shap_rank = np.argsort(-global_shap_importance)
    top10_global = [(feature_names[i], float(global_shap_importance[i]))
                      for i in global_shap_rank[:10]]

    # Figure 3: global SHAP bar chart
    plt.figure(figsize=(7, 5))
    names_plot = [n for n, _ in top10_global][::-1]
    vals_plot = [v for _, v in top10_global][::-1]
    plt.barh(names_plot, vals_plot, color="#4472C4")
    plt.xlabel("Mean |SHAP value| (n=50 held-out test instances)")
    plt.title("Global feature attributions (tuned Random Forest)")
    plt.tight_layout()
    plt.savefig("outputs/figure3_global_shap.png", dpi=300)
    plt.close()

    # Case study: lowest predicted pass probability among the 50 sampled
    case_local_i = int(np.argmin(proba_test[global_idx]))
    case_i = int(global_idx[case_local_i])
    case_proba = float(proba_test[case_i])
    case_shap = shap_matrix[case_local_i]
    case_lime = explainer.lime(X_test_arr[case_i])
    case_jaccard = top_k_jaccard(case_shap, case_lime, feature_names, k=5)

    top_union = sorted(
        set(np.argsort(-np.abs(case_shap))[:5]) | set(np.argsort(-np.abs(case_lime))[:5]),
        key=lambda j: -abs(case_shap[j])
    )
    table6 = [{"feature": feature_names[j], "shap": float(case_shap[j]),
               "lime": float(case_lime[j])} for j in top_union]

    # Systematic 50-instance agreement
    jaccards, spearmans = [], []
    for local_i, i in enumerate(global_idx):
        shap_i = shap_matrix[local_i]
        lime_i = explainer.lime(X_test_arr[i])
        jaccards.append(top_k_jaccard(shap_i, lime_i, feature_names, k=5))
        rho, _ = stats.spearmanr(np.abs(shap_i), np.abs(lime_i))
        spearmans.append(rho)
    mean_jaccard = float(np.mean(jaccards))
    mean_spearman = float(np.nanmean(spearmans))

    # Permutation importance
    perm = permutation_importance(rf, X_test_arr, y_test_arr, n_repeats=30,
                                   random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1)
    perm_importance = perm.importances_mean
    rho_perm, p_perm = stats.spearmanr(global_shap_importance, perm_importance)
    perm_rank = np.argsort(-perm_importance)[:10]
    top10_perm = [(feature_names[i], float(perm_importance[i])) for i in perm_rank]

    report = {
        "n_total": int(X.shape[0]), "n_features": int(X.shape[1]),
        "class_distribution": {"fail": int((y == 0).sum()), "pass": int((y == 1).sum())},
        "table2_hyperparameters": table2,
        "table3_repeated_cv": table3,
        "significance": sig,
        "illustrative_split": illus,
        "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
        "top10_global_shap": top10_global,
        "case_study": {"p_pass": case_proba, "table6": table6, "jaccard": case_jaccard},
        "systematic_agreement": {"mean_jaccard": mean_jaccard, "mean_spearman": mean_spearman},
        "top10_permutation_importance": top10_perm,
        "spearman_shap_vs_permutation": {"rho": float(rho_perm), "p": float(p_perm)},
        "random_seed": RANDOM_SEED,
        "n_train": len(y_train), "n_test": len(y_test),
    }
    with open("outputs/full_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
