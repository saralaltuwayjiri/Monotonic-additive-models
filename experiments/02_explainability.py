"""
Reproduces the manuscript's explainability results: global SHAP
attributions (Figure 3 / Section 4.5), the single-instance case study
(Table 6 / Section 4.6), the systematic 50-instance SHAP-LIME agreement
study (Section 4.6), and the permutation-importance cross-check
(Section 4.7).

Uses the tuned Random Forest model fit on the 80/20 training split
(same split as the illustrative evaluation), applied to the held-out
test instances.

Usage: python3 experiments/02_explainability.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.preprocess import load_and_preprocess
from src.models import tune_hyperparameters, build_models, RANDOM_SEED
from src.shap_lime_impl import ShapLimeExplainer


def top_k_jaccard(attr_a, attr_b, feature_names, k=5):
    top_a = set(np.array(feature_names)[np.argsort(-np.abs(attr_a))[:k]])
    top_b = set(np.array(feature_names)[np.argsort(-np.abs(attr_b))[:k]])
    return len(top_a & top_b) / len(top_a | top_b)


def main():
    X, y, feature_names = load_and_preprocess("data/student-mat.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    tuned = tune_hyperparameters(X_train, y_train)
    models = build_models(tuned)
    rf = models["rf"]
    rf.fit(X_train.values, y_train)

    predict_fn = lambda Z: rf.predict_proba(Z)[:, 1]

    explainer = ShapLimeExplainer(
        predict_fn, X_train.values, n_features=X.shape[1],
        seed=RANDOM_SEED, background_size=50,
    )

    X_test_arr = X_test.values
    y_test_arr = y_test.values
    proba_test = predict_fn(X_test_arr)

    # ---- Global SHAP attributions over held-out test instances ----
    n_global = min(50, len(X_test_arr))
    rng = np.random.RandomState(RANDOM_SEED)
    global_idx = rng.choice(len(X_test_arr), n_global, replace=False)

    shap_matrix = np.array([
        explainer.kernel_shap(X_test_arr[i]) for i in global_idx
    ])
    global_shap_importance = np.abs(shap_matrix).mean(axis=0)
    global_shap_rank = np.argsort(-global_shap_importance)
    top10_global = [(feature_names[i], float(global_shap_importance[i]))
                     for i in global_shap_rank[:10]]

    print("--- Top 10 global SHAP attributions (mean |phi|, n=50 test instances) ---")
    for name, val in top10_global:
        print(f"  {name}: {val:.4f}")

    # ---- Case study: lowest predicted pass probability among the 50 ----
    case_local_i = int(np.argmin(proba_test[global_idx]))
    case_i = int(global_idx[case_local_i])
    case_proba = float(proba_test[case_i])
    case_shap = shap_matrix[case_local_i]
    case_lime = explainer.lime(X_test_arr[case_i])

    case_shap_rank = np.argsort(-np.abs(case_shap))[:5]
    case_lime_rank = np.argsort(-np.abs(case_lime))[:5]
    case_table = [
        {"feature": feature_names[j], "shap": float(case_shap[j]),
         "lime": float(case_lime[j])}
        for j in sorted(set(case_shap_rank) | set(case_lime_rank),
                         key=lambda j: -abs(case_shap[j]))
    ]
    case_jaccard = top_k_jaccard(case_shap, case_lime, feature_names, k=5)

    print(f"\n--- Case study: held-out student, predicted P(pass)={case_proba:.3f} ---")
    for row in case_table:
        print(f"  {row['feature']}: SHAP={row['shap']:.4f}  LIME={row['lime']:.4f}")
    print(f"  top-5 Jaccard overlap for this instance: {case_jaccard:.3f}")

    # ---- Systematic 50-instance SHAP-LIME agreement ----
    jaccards, spearmans = [], []
    lime_matrix = []
    for local_i, i in enumerate(global_idx):
        shap_i = shap_matrix[local_i]
        lime_i = explainer.lime(X_test_arr[i])
        lime_matrix.append(lime_i)
        jaccards.append(top_k_jaccard(shap_i, lime_i, feature_names, k=5))
        rho, _ = stats.spearmanr(np.abs(shap_i), np.abs(lime_i))
        spearmans.append(rho)
    lime_matrix = np.array(lime_matrix)

    mean_jaccard = float(np.mean(jaccards))
    mean_spearman = float(np.nanmean(spearmans))

    print(f"\n--- Systematic agreement over {n_global} held-out instances ---")
    print(f"  mean top-5 Jaccard similarity: {mean_jaccard:.3f}")
    print(f"  mean Spearman rank correlation (|SHAP| vs |LIME|): {mean_spearman:.3f}")

    # ---- Permutation importance cross-check ----
    perm = permutation_importance(
        rf, X_test_arr, y_test_arr, n_repeats=30,
        random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1,
    )
    perm_importance = perm.importances_mean
    rho_perm, p_perm = stats.spearmanr(global_shap_importance, perm_importance)

    perm_rank = np.argsort(-perm_importance)[:10]
    print("\n--- Top 10 permutation importance (30 repeats, scored on test ROC-AUC) ---")
    for i in perm_rank:
        print(f"  {feature_names[i]}: {perm_importance[i]:.4f}")
    print(f"\n  Spearman(SHAP global, permutation importance) = {rho_perm:.3f}, p={p_perm:.4g}")

    out = {
        "n_test": int(len(X_test_arr)),
        "n_global_instances": n_global,
        "top10_global_shap": top10_global,
        "case_study": {
            "instance_index_in_test": case_i,
            "predicted_p_pass": case_proba,
            "table": case_table,
            "top5_jaccard": case_jaccard,
        },
        "systematic_agreement": {
            "mean_top5_jaccard": mean_jaccard,
            "mean_spearman_abs_shap_lime": mean_spearman,
        },
        "permutation_importance": {
            "top10": [(feature_names[i], float(perm_importance[i])) for i in perm_rank],
            "spearman_vs_shap": float(rho_perm),
            "spearman_vs_shap_p": float(p_perm),
        },
        "random_seed": RANDOM_SEED,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/explainability_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved outputs/explainability_results.json")


if __name__ == "__main__":
    main()
