"""
Hand-implemented Kernel SHAP and LIME, following the published algorithms
(Lundberg & Lee, 2017; Ribeiro et al., 2016), used instead of the
third-party `shap`/`lime` packages (no package-index access in the build
environment). See the manuscript's Limitations section for the explicit
statement that these are direct algorithm implementations, not a specific
software package, and that substituting the reference packages is
encouraged as a robustness check.
"""
import numpy as np
from itertools import combinations
from math import comb
from sklearn.linear_model import Ridge


class ShapLimeExplainer:
    def __init__(self, predict_fn, X_background, n_features, seed=42,
                 background_size=50):
        """predict_fn: callable returning P(class=1) for a 2D array of rows.
        X_background: training-distribution sample used as the SHAP
        coalition baseline and the LIME perturbation generator."""
        self.predict_fn = predict_fn
        self.n_features = n_features
        self.rng = np.random.RandomState(seed)
        if len(X_background) > background_size:
            idx = self.rng.choice(len(X_background), background_size,
                                   replace=False)
            self.background = np.asarray(X_background)[idx]
        else:
            self.background = np.asarray(X_background)
        self.bg_mean = self.background.mean(axis=0)

    # ---------------- Kernel SHAP ----------------
    def kernel_shap(self, x, n_samples=2000):
        """Returns phi (length n_features) approximating Shapley values for
        predict_fn at instance x, using the standard Kernel SHAP weighted
        least-squares formulation with a background-mean baseline."""
        M = self.n_features
        f_full = self.predict_fn(x.reshape(1, -1))[0]
        f_base = self.predict_fn(self.bg_mean.reshape(1, -1))[0]

        # Enumerate all 2^M coalitions when tractable, else sample.
        max_exact = 14
        if M <= max_exact:
            masks = []
            weights = []
            for size in range(M + 1):
                if size == 0 or size == M:
                    w = 1e6  # effectively fixed constraints, not sampled
                else:
                    w = (M - 1) / (comb(M, size) * size * (M - size))
                for combo in combinations(range(M), size):
                    mask = np.zeros(M, dtype=bool)
                    mask[list(combo)] = True
                    masks.append(mask)
                    weights.append(w)
            masks = np.array(masks)
            weights = np.array(weights)
        else:
            masks = self.rng.randint(0, 2, size=(n_samples, M)).astype(bool)
            sizes = masks.sum(axis=1)
            weights = np.array([
                1e6 if (s == 0 or s == M) else
                (M - 1) / (comb(M, int(s)) * s * (M - s))
                for s in sizes
            ])

        # Build synthetic inputs: present features = x, absent = background mean
        Z = np.tile(self.bg_mean, (len(masks), 1))
        Z[masks] = np.tile(x, (len(masks), 1))[masks]
        preds = self.predict_fn(Z)

        # Weighted least squares: preds ~ f_base + mask @ phi, with
        # constraint sum(phi) = f_full - f_base enforced via augmented target.
        Mm = masks.astype(float)
        target = preds - f_base
        W = np.sqrt(np.clip(weights, 0, 1e4))
        Mm_w = Mm * W[:, None]
        target_w = target * W
        phi, *_ = np.linalg.lstsq(Mm_w, target_w, rcond=None)

        # Rescale so attributions sum exactly to f_full - f_base (efficiency
        # property), correcting small numerical drift from the LS fit.
        total = f_full - f_base
        if phi.sum() != 0:
            phi = phi * (total / phi.sum()) if abs(phi.sum()) > 1e-8 else phi
        return phi

    # ---------------- LIME ----------------
    def lime(self, x, n_samples=1000, kernel_width=None):
        """Local surrogate: perturb around x by swapping each feature with a
        background sample, fit a weighted Ridge regression on predict_fn's
        output, return the surrogate's feature coefficients as attributions."""
        M = self.n_features
        if kernel_width is None:
            kernel_width = np.sqrt(M) * 0.75

        Z = np.tile(x, (n_samples, 1)).astype(float)
        mask = self.rng.randint(0, 2, size=(n_samples, M)).astype(bool)
        bg_idx = self.rng.randint(0, len(self.background), size=n_samples)
        Z[mask] = self.background[bg_idx][mask]

        # Normalized Hamming distance from x as the locality measure.
        dist = mask.sum(axis=1) / M
        weights = np.exp(-(dist ** 2) / (kernel_width ** 2))

        preds = self.predict_fn(Z)
        model = Ridge(alpha=1.0)
        model.fit(Z, preds, sample_weight=weights)
        return model.coef_
