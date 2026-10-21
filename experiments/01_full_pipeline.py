"""
Reproduces the manuscript's Section 4 results: hyperparameter search
(Table 2), repeated cross-validation primary metric (Table 3), and the
illustrative single-split evaluation (Section 4.4).

Usage: python3 experiments/01_full_pipeline.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import load_and_preprocess
from src.models import (
    tune_hyperparameters, build_models, repeated_cv_scores,
    paired_tests, illustrative_split_eval, RANDOM_SEED,
)
from sklearn.model_selection import train_test_split

def main():
    X, y, feature_names = load_and_preprocess("data/student-mat.csv")
    print(f"Loaded {X.shape[0]} students, {X.shape[1]} features.")

    # 80/20 split used ONLY for hyperparameter search + illustrative eval
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    print("\n--- Tuning hyperparameters (RandomizedSearchCV, 5-fold CV on train) ---")
    tuned = tune_hyperparameters(X_train, y_train)
    for name, info in tuned.items():
        print(f"  {name}: {info['best_params']}")

    models = build_models(tuned)

    print("\n--- Primary metric: repeated 5x5 stratified CV on FULL dataset ---")
    scores = repeated_cv_scores(models, X, y, tuned["lr"]["scaler"])
    summary = {}
    for name, s in scores.items():
        summary[name] = {"mean": float(s.mean()), "std": float(s.std())}
        print(f"  {name}: mean ROC-AUC = {s.mean():.3f} +/- {s.std():.3f}")

    print("\n--- Paired significance tests across the 25 shared folds ---")
    sig = {
        "rf_vs_gb": paired_tests(scores["rf"], scores["gb"]),
        "rf_vs_lr": paired_tests(scores["rf"], scores["lr"]),
        "lr_vs_gb": paired_tests(scores["lr"], scores["gb"]),
    }
    for pair, res in sig.items():
        print(f"  {pair}: mean_diff={res['mean_diff']:.4f}, "
              f"paired t={res['t_stat']:.2f}, p={res['t_p']:.4g}, "
              f"wilcoxon p={res['wilcoxon_p']:.4g}")

    print("\n--- Illustrative single 80/20 split (Section 4.4) ---")
    illus, split_data = illustrative_split_eval(models, X, y, tuned["lr"]["scaler"])
    for name, r in illus.items():
        print(f"  {name}: test AUC={r['test_auc']:.3f}, "
              f"confusion_matrix={r['confusion_matrix']}")

    out = {
        "n_total": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_distribution": {
            "fail": int((y == 0).sum()), "pass": int((y == 1).sum())
        },
        "tuned_hyperparameters": {k: v["best_params"] for k, v in tuned.items()},
        "repeated_cv_summary": summary,
        "paired_significance": sig,
        "illustrative_split": illus,
        "random_seed": RANDOM_SEED,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/pipeline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved outputs/pipeline_results.json")

if __name__ == "__main__":
    main()
