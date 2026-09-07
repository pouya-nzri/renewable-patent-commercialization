"""
modeling.py
===========
Learners, Optuna tuning, leakage-safe cross-validation, and grouped SHAP.

Key correctness points versus the original script:
  * Structured features are standardized with TRAIN-fold moments only
    (StandardScaler fit inside each fold) -> no leakage.              [paper]
  * Grouped SHAP: a TreeExplainer computes per-instance SHAP values;
    absolute values are summed within each feature group's column span
    and normalized to a within-run share Φ̄_g.                        [paper]
  * Every run yields (rmse, r2, group_share_vector), the atoms the
    aggregation layer needs to reproduce Tables 6-8 and Figure 5.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import config as C

# Gradient-boosting backends
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    _HAS_CB = True
except Exception:
    _HAS_CB = False


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def make_model(learner: str, params: dict | None = None):
    params = dict(params or {})
    if learner == "lightgbm":
        params.setdefault("objective", "regression")
        params.setdefault("random_state", C.RANDOM_SEED)
        return lgb.LGBMRegressor(**params, verbose=-1, n_jobs=-1)
    if learner == "xgboost":
        params.setdefault("objective", "reg:squarederror")
        params.setdefault("random_state", C.RANDOM_SEED)
        return xgb.XGBRegressor(**params, verbosity=0, n_jobs=-1)
    if learner == "catboost":
        if not _HAS_CB:
            raise RuntimeError("catboost not installed")
        params.setdefault("random_seed", C.RANDOM_SEED)
        return cb.CatBoostRegressor(**params, verbose=False)
    raise ValueError(f"Unknown learner '{learner}'")


# --------------------------------------------------------------------------- #
# Optuna search spaces (unchanged ranges from the user's script, seeded)
# --------------------------------------------------------------------------- #
def _suggest(trial, learner: str) -> dict:
    if learner == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", -1, 50),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if learner == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
    if learner == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    raise ValueError(learner)


def _scale_in_fold(X_tr, X_val, n_structured: int):
    """
    Standardize ONLY the structured columns, using train moments.  [paper: z_ij]
    Embedding columns are left as-is (already in a learned metric space).
    """
    scaler = StandardScaler()
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_tr[:, :n_structured] = scaler.fit_transform(X_tr[:, :n_structured])
    X_val[:, :n_structured] = scaler.transform(X_val[:, :n_structured])
    return X_tr, X_val, scaler


def tune(X, y, learner: str, n_trials: int = C.N_OPTUNA_TRIALS,
         n_structured: int = len(C.STRUCTURED_FIELDS)) -> dict:
    """Optuna TPE tuning under 10-fold CV, minimizing mean RMSE."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    kf = KFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.RANDOM_SEED)

    def objective(trial):
        params = _suggest(trial, learner)
        rmses = []
        for tr, val in kf.split(X):
            X_tr, X_val, _ = _scale_in_fold(X[tr], X[val], n_structured)
            m = make_model(learner, params)
            m.fit(X_tr, y[tr])
            pred = m.predict(X_val)
            rmses.append(np.sqrt(mean_squared_error(y[val], pred)))
        return float(np.mean(rmses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=C.RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# --------------------------------------------------------------------------- #
# Grouped SHAP
# --------------------------------------------------------------------------- #
def grouped_shap_share(model, X_sample: np.ndarray,
                       group_slices: Dict[str, tuple]) -> Dict[str, float]:
    """
    Compute the within-run normalized SHAP share for each feature group.

    Steps (paper, Section 3):
      1. TreeExplainer -> per-instance, per-column SHAP values.
      2. For each group g, sum mean(|SHAP|) over its columns D_g.
      3. Normalize across groups so shares sum to 1 (=100%).
    Returns {group_name: share in [0,1]}.
    """
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):          # some backends return a list
        sv = sv[0]
    mean_abs = np.abs(sv).mean(axis=0)             # per-column mean |SHAP|

    raw = {}
    for g, (a, b) in group_slices.items():
        raw[g] = float(mean_abs[a:b].sum())
    total = sum(raw.values()) or 1.0
    return {g: v / total for g, v in raw.items()}


# --------------------------------------------------------------------------- #
# One evaluated run = one (encoder, learner, dataset, fold)
# --------------------------------------------------------------------------- #
def evaluate_cv(X, y, learner: str, params: dict,
                group_slices: Dict[str, tuple],
                shap_max_instances: int = 500,
                n_structured: int = len(C.STRUCTURED_FIELDS)) -> List[dict]:
    """
    Run 10-fold CV. For each fold return a record with rmse, r2, and the
    grouped SHAP share vector, i.e. one row per model-dataset-fold run.
    """
    kf = KFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.RANDOM_SEED)
    records = []
    for fold, (tr, val) in enumerate(kf.split(X)):
        X_tr, X_val, _ = _scale_in_fold(X[tr], X[val], n_structured)
        model = make_model(learner, params)
        model.fit(X_tr, y[tr])
        pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y[val], pred)))
        r2 = float(r2_score(y[val], pred))

        # grouped SHAP on a capped validation subsample (for tractability)
        idx = np.arange(X_val.shape[0])
        if len(idx) > shap_max_instances:
            rng = np.random.RandomState(C.RANDOM_SEED + fold)
            idx = rng.choice(idx, shap_max_instances, replace=False)
        shares = grouped_shap_share(model, X_val[idx], group_slices)

        records.append({
            "fold": fold, "rmse": rmse, "r2": r2, "shares": shares, "model": model,
        })
    return records
