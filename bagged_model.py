"""
bagged_model.py — the BaggedEnsemble container.

Lives at project root (not inside validation/) so that pickled bags resolve to a stable,
importable module. Trained by validation/train_bagged.py; used by app.py as an uncertainty
companion: the spread across members is a real variance estimate for one molecule.
"""
import numpy as np


class BaggedEnsemble:
    """
    Bag of independently-seeded regressors fitted on bootstrap resamples.

    Exposes `.predict()` (the bag mean) and `.estimators_` (the members), so it is a drop-in
    both for a normal scikit-learn-style model and for app.py's "ensemble" uncertainty path.
    """

    def __init__(self, estimators, prop=None, meta=None):
        self.estimators_ = list(estimators)
        self.prop = prop
        self.meta = meta or {}

    def predict(self, X):
        return np.mean([e.predict(X) for e in self.estimators_], axis=0)

    def predict_with_std(self, X):
        """(mean, std) across bag members -- the std is the per-prediction uncertainty."""
        P = np.array([e.predict(X) for e in self.estimators_])
        return P.mean(axis=0), P.std(axis=0)

    def __len__(self):
        return len(self.estimators_)

    def __repr__(self):
        return f"BaggedEnsemble(prop={self.prop!r}, n={len(self.estimators_)})"
