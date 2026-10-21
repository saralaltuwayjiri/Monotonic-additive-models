"""
Preprocessing for the UCI Student Performance (Mathematics) dataset.

Source: Cortez & Silva (2008), UCI Machine Learning Repository.
Dataset DOI: https://doi.org/10.24432/C5TG7T
Dataset URL: https://archive.ics.uci.edu/dataset/320/student+performance

Implements exactly the protocol described in the manuscript's
"Preprocessing and Encoding Protocol" subsection:
  - Binary yes/no attributes mapped to a 0/1 indicator.
  - Nominal variables with exactly two levels encoded as a single
    reference-level dummy column (drop_first).
  - Nominal variables with more than two levels one-hot encoded,
    also dropping one reference level per variable (drop_first),
    which reproduces the reported final feature count of 39.
  - G1 and G2 dropped from the feature matrix to avoid target leakage.
  - Target: Y = 1 if G3 >= 10 (Pass), else 0.
"""
import pandas as pd

BINARY_YESNO = ["schoolsup", "famsup", "paid", "activities",
                 "nursery", "higher", "internet", "romantic"]
TWO_LEVEL_NOMINAL = ["school", "sex", "address", "famsize", "Pstatus"]
MULTI_LEVEL_NOMINAL = ["Mjob", "Fjob", "reason", "guardian"]
NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
           "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]

# Features given a monotonic/behavioral interpretation in the paper's
# discussion of global attributions; not used to constrain any model here
# (no monotonicity constraint is applied in this manuscript's pipeline),
# listed only for documentation of feature semantics.
FEATURE_NOTES = {
    "failures": "number of past class failures (0-3)",
    "absences": "number of school absences",
    "studytime": "weekly study time (1-4, ordinal)",
}


def load_and_preprocess(csv_path: str):
    """Load student-mat.csv and return (X, y, feature_names)."""
    df = pd.read_csv(csv_path, sep=";")

    y = (df["G3"] >= 10).astype(int)

    X = df.drop(columns=["G1", "G2", "G3"]).copy()

    for col in BINARY_YESNO:
        X[col] = (X[col] == "yes").astype(int)

    X = pd.get_dummies(
        X, columns=TWO_LEVEL_NOMINAL + MULTI_LEVEL_NOMINAL, drop_first=True
    )

    # sklearn/pandas may emit boolean dtype columns from get_dummies; cast all
    # to float64 for compatibility with every estimator used downstream.
    X = X.astype("float64")

    return X, y, list(X.columns)


if __name__ == "__main__":
    X, y, names = load_and_preprocess("data/student-mat.csv")
    print(f"X shape: {X.shape}")
    print(f"Feature count: {X.shape[1]} (manuscript reports 39)")
    print(f"Class distribution: Fail={int((y == 0).sum())} "
          f"({(y == 0).mean()*100:.1f}%), "
          f"Pass={int((y == 1).sum())} ({(y == 1).mean()*100:.1f}%)")
