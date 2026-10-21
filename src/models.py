"""
Model specification, hyperparameter search, and repeated cross-validation
exactly as described in the manuscript's "Model Specification and
Hyperparameter Search" and "Evaluation Protocol" subsections.

Three paradigms are compared:
  - Logistic Regression (LR): structurally transparent baseline.
  - Random Forest (RF, Breiman 2001): bagging tree ensemble.
  - Histogram-based Gradient Boosting (HistGradientBoostingClassifier):
    implemented with the same core algorithmic ingredients as XGBoost
    (binned feature histograms, shrinkage, L2-regularized leaf weights).

RandomizedSearchCV is run on the 80% training split only (5-fold
stratified CV, scoring = ROC-AUC). The resulting tuned hyperparameters
are then evaluated with repeated stratified 5-fold CV (5 repeats, 25
folds, fixed seed shared across models) on the FULL dataset, which is
the manuscript's primary performance metric.
"""
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def tune_hyperparameters(X_train, y_train, scale_for_lr=True):
    """RandomizedSearchCV for LR, RF, HistGB on the training split only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr_param_dist = {"C": np.logspace(-3, 2, 7)}
    lr_search = RandomizedSearchCV(
        LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        lr_param_dist, n_iter=7, scoring="roc_auc", cv=cv,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    lr_search.fit(X_train_scaled, y_train)

    rf_param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 4, 6, 8, 10, 12],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "max_features": ["sqrt", "log2", 0.5, 0.7, None],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED),
        rf_param_dist, n_iter=25, scoring="roc_auc", cv=cv,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)

    gb_param_dist = {
        "max_iter": [50, 100, 150, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "max_depth": [None, 3, 4, 5, 6],
        "max_leaf_nodes": [15, 31, 63, 127],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0],
    }
    gb_search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=RANDOM_SEED),
        gb_param_dist, n_iter=25, scoring="roc_auc", cv=cv,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    gb_search.fit(X_train, y_train)

    return {
        "lr": {"best_params": lr_search.best_params_, "scaler": scaler,
               "best_cv_score": float(lr_search.best_score_)},
        "rf": {"best_params": rf_search.best_params_,
               "best_cv_score": float(rf_search.best_score_)},
        "gb": {"best_params": gb_search.best_params_,
               "best_cv_score": float(gb_search.best_score_)},
    }


def build_models(tuned):
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED,
                             **tuned["lr"]["best_params"])
    rf = RandomForestClassifier(random_state=RANDOM_SEED,
                                 **tuned["rf"]["best_params"])
    gb = HistGradientBoostingClassifier(random_state=RANDOM_SEED,
                                         **tuned["gb"]["best_params"])
    return {"lr": lr, "rf": rf, "gb": gb}


def repeated_cv_scores(models, X, y, scaler):
    """5x5 repeated stratified CV ROC-AUC on the full dataset, same fold
    partitions (fixed seed) shared across all three models."""
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                   random_state=RANDOM_SEED)
    scores = {}
    X_scaled = scaler.transform(X)
    for name, model in models.items():
        X_use = X_scaled if name == "lr" else X.values
        s = cross_val_score(model, X_use, y, cv=rcv, scoring="roc_auc",
                             n_jobs=-1)
        scores[name] = s
    return scores


def paired_tests(scores_a, scores_b):
    diffs = scores_a - scores_b
    t_stat, t_p = stats.ttest_rel(scores_a, scores_b)
    w_stat, w_p = stats.wilcoxon(scores_a, scores_b)
    return {
        "mean_diff": float(diffs.mean()),
        "t_stat": float(t_stat), "t_p": float(t_p),
        "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
    }


def illustrative_split_eval(models, X, y, scaler):
    """Single fixed 80/20 stratified split, illustrative only (Section 4.4)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    results = {}
    for name, model in models.items():
        if name == "lr":
            Xtr = scaler.transform(X_train)
            Xte = scaler.transform(X_test)
        else:
            Xtr, Xte = X_train.values, X_test.values
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        from sklearn.metrics import roc_auc_score, confusion_matrix
        auc = roc_auc_score(y_test, proba)
        preds = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds)
        results[name] = {"test_auc": float(auc), "confusion_matrix": cm.tolist(),
                          "n_train": len(y_train), "n_test": len(y_test)}
    return results, (X_train, X_test, y_train, y_test)
