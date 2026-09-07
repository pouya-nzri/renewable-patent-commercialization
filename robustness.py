"""
robustness.py
=============
All robustness and validation analyses described in the paper's Results:

  * ablation ladder (a-f): structured-only, TF-IDF+SVD, Doc2Vec, SciBERT,
    PatentBERT, general text-only (BERT/ELECTRA), hybrid       -> Ablation table
  * memorization baselines: one-hot ID, hashed ID, portfolio-frequency proxy
  * cold-start: owner-disjoint folds (GroupKFold on assignee)  -> R^2=0.939 claim
  * temporal-accumulation: Pearson r(patent age, target) per stratum -> age table
  * LIFT / cumulative-gain analysis vs bibliometric baseline   -> lift figure
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher

import config as C
from modeling import make_model, _scale_in_fold


# --------------------------------------------------------------------------- #
# Generic CV scorer (RMSE, R2) with in-fold structured scaling
# --------------------------------------------------------------------------- #
def cv_score(X, y, learner="lightgbm", params=None,
             n_structured=0, groups=None) -> Tuple[float, float]:
    if groups is not None:
        splitter = GroupKFold(n_splits=C.N_SPLITS)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = KFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.RANDOM_SEED)
        split_iter = splitter.split(X)

    rmses, r2s = [], []
    for tr, val in split_iter:
        if n_structured > 0:
            X_tr, X_val, _ = _scale_in_fold(X[tr], X[val], n_structured)
        else:
            X_tr, X_val = X[tr], X[val]
        m = make_model(learner, params or _default_params(learner))
        m.fit(X_tr, y[tr])
        pred = m.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y[val], pred)))
        r2s.append(r2_score(y[val], pred))
    return float(np.mean(rmses)), float(np.mean(r2s))


def _default_params(learner: str) -> dict:
    # modest, fixed defaults for baselines (tuning reserved for the main models)
    if learner == "lightgbm":
        return {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63}
    if learner == "xgboost":
        return {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6}
    return {"iterations": 400, "learning_rate": 0.05, "depth": 6}


# --------------------------------------------------------------------------- #
# 1) Ablation ladder on Dataset-3
# --------------------------------------------------------------------------- #
def ablation_ladder(df, struct: np.ndarray, y: np.ndarray,
                    field_emb_general: Dict[str, Dict[str, np.ndarray]],
                    field_emb_domain: Dict[str, Dict[str, np.ndarray]] | None = None,
                    learner: str = "lightgbm") -> "pd.DataFrame":
    """
    Build the six-regime ablation table on Dataset-3.

    field_emb_general: {"bert": {field:emb}, "electra": {field:emb}}
    field_emb_domain : {"scibert": {...}, "patentbert": {...}} (optional)
    """
    import pandas as pd
    ns = struct.shape[1]
    rows = []

    def concat_text(emb_dict):
        return np.concatenate([emb_dict[f] for f in C.TEXT_FIELDS], axis=1)

    # (a) structured-only
    rmse, r2 = cv_score(struct, y, learner, n_structured=ns)
    rows.append(("Structured indicators only", "Structured (counts)", rmse, r2))

    # (b) TF-IDF + truncated SVD over concatenated cleaned text
    corpus = (df["assignees_text"] + " " + df["claims_text"] + " " +
              df["countries_text"] + " " + df["backward_refs_text"] + " " +
              df["inventors_text"] + " " + df["abstract_text"]).tolist()
    tfidf = TfidfVectorizer(max_features=5000)
    Xtf = tfidf.fit_transform(corpus)
    svd = TruncatedSVD(n_components=300, random_state=C.RANDOM_SEED)
    Xtf = svd.fit_transform(Xtf)
    rmse, r2 = cv_score(Xtf, y, learner, n_structured=0)
    rows.append(("TF-IDF (+ truncated SVD)", "Text (sparse lexical)", rmse, r2))

    # (c) Doc2Vec (static dense) -- optional dependency
    try:
        Xd2v = _doc2vec_matrix(corpus)
        rmse, r2 = cv_score(Xd2v, y, learner, n_structured=0)
        rows.append(("Doc2Vec", "Text (static dense)", rmse, r2))
    except Exception as e:
        print(f"[ablation] Doc2Vec skipped ({e}); install gensim to enable.")

    # (d) domain-specific contextual text-only
    if field_emb_domain:
        for name in ("scibert", "patentbert"):
            if name in field_emb_domain:
                Xtxt = concat_text(field_emb_domain[name])
                rmse, r2 = cv_score(Xtxt, y, learner, n_structured=0)
                label = "SciBERT [CLS]" if name == "scibert" else "PatentBERT [CLS]"
                rows.append((label, "Text (contextual, domain)", rmse, r2))

    # (e) general contextual text-only (BERT, ELECTRA)
    for name in ("bert", "electra"):
        if name in field_emb_general:
            Xtxt = concat_text(field_emb_general[name])
            rmse, r2 = cv_score(Xtxt, y, learner, n_structured=0)
            rows.append((f"{name.upper()}-base [CLS]", "Text (contextual, general)", rmse, r2))

    # (f) hybrid: best general encoder + structured
    best = "electra" if "electra" in field_emb_general else "bert"
    Xhy = np.concatenate([struct, concat_text(field_emb_general[best])], axis=1)
    rmse, r2 = cv_score(Xhy, y, learner, n_structured=ns)
    rows.append((f"{best.upper()}-base + structured", "Hybrid (best)", rmse, r2))

    return pd.DataFrame(rows, columns=["Representation", "Modality", "RMSE", "R2"])


def _doc2vec_matrix(corpus: List[str]) -> np.ndarray:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    docs = [TaggedDocument(words=t.split(), tags=[str(i)]) for i, t in enumerate(corpus)]
    model = Doc2Vec(docs, vector_size=300, window=5, min_count=2,
                    workers=4, epochs=20, seed=C.RANDOM_SEED)
    return np.vstack([model.dv[str(i)] for i in range(len(corpus))])


# --------------------------------------------------------------------------- #
# 2) Memorization baselines (identity encodings of the assignee field)
# --------------------------------------------------------------------------- #
def memorization_baselines(df, struct: np.ndarray, y: np.ndarray,
                           assignee_emb: np.ndarray,
                           learner: str = "lightgbm") -> "pd.DataFrame":
    """
    Compare the 768-d assignee embedding against bare-identity encodings:
      * one-hot / label-encoded assignee id
      * hashed assignee id
      * portfolio-frequency (size) proxy
    All hold the other features fixed (here: structured + assignee channel).
    """
    import pandas as pd
    ns = struct.shape[1]
    keys = df["assignee_key"].values
    rows = []

    # proposed: structured + assignee embedding
    Xemb = np.concatenate([struct, assignee_emb], axis=1)
    rmse, r2 = cv_score(Xemb, y, learner, n_structured=ns)
    rows.append(("768-d name embedding (proposed)", "Contextual name semantics", rmse, r2))

    # one-hot / label id
    from sklearn.preprocessing import LabelEncoder
    lab = LabelEncoder().fit_transform(keys).reshape(-1, 1).astype(float)
    Xlab = np.concatenate([struct, lab], axis=1)
    rmse, r2 = cv_score(Xlab, y, learner, n_structured=ns)
    rows.append(("One-hot / label-encoded assignee ID", "Bare identity", rmse, r2))

    # hashed id
    hasher = FeatureHasher(n_features=64, input_type="string")
    Xhash = hasher.transform([[k] for k in keys]).toarray()
    Xh = np.concatenate([struct, Xhash], axis=1)
    rmse, r2 = cv_score(Xh, y, learner, n_structured=ns)
    rows.append(("Hashed assignee identifier", "Bare identity (hashed)", rmse, r2))

    # portfolio-frequency proxy
    freq = df.groupby("assignee_key")["assignee_key"].transform("count").values.reshape(-1, 1).astype(float)
    Xf = np.concatenate([struct, freq], axis=1)
    rmse, r2 = cv_score(Xf, y, learner, n_structured=ns + 1)
    rows.append(("Assignee portfolio frequency", "Size proxy", rmse, r2))

    return pd.DataFrame(rows, columns=["Representation", "Encoding", "RMSE", "R2"])


# --------------------------------------------------------------------------- #
# 3) Cold-start: owner-disjoint folds
# --------------------------------------------------------------------------- #
def cold_start(df, X: np.ndarray, y: np.ndarray,
               learner: str = "lightgbm",
               n_structured: int = len(C.STRUCTURED_FIELDS)) -> Dict[str, float]:
    """Owner-disjoint evaluation: every test assignee is unseen in training."""
    groups = df["assignee_key"].values
    rmse, r2 = cv_score(X, y, learner, n_structured=n_structured, groups=groups)
    return {"cold_start_rmse": rmse, "cold_start_r2": r2}


# --------------------------------------------------------------------------- #
# 4) Temporal-accumulation: patent age vs target
# --------------------------------------------------------------------------- #
def age_correlation(df, y: np.ndarray, current_year: int = 2024) -> Dict[str, float]:
    """
    Pearson r between patent age and the target, and the implied share of
    target variance (r^2). Age is derived from a grant/filing year field if
    present; otherwise this returns NaN and should be supplied by the author.
    """
    from scipy import stats
    year_col = None
    for cand in ("grant_year", "filing_year", "year"):
        if cand in df.columns:
            year_col = cand
            break
    if year_col is None:
        return {"pearson_r": float("nan"), "variance_share": float("nan"),
                "note": "no year field found; add grant_year to enable"}
    age = current_year - df[year_col].astype(float).values
    r, p = stats.pearsonr(age, y)
    return {"pearson_r": float(r), "p_value": float(p),
            "variance_share": float(r ** 2)}


# --------------------------------------------------------------------------- #
# 5) LIFT / cumulative-gain
# --------------------------------------------------------------------------- #
def lift_curve(y_true: np.ndarray, y_score: np.ndarray,
               top_decile_as_positive: bool = True) -> Dict[str, object]:
    """
    Cumulative-gain curve: sort by score descending; at each decile of the
    screened portfolio, report the fraction of true high-frequency assets
    captured. High-frequency = top decile of realized target.
    """
    n = len(y_true)
    thresh = np.quantile(y_true, 0.9)
    positive = y_true >= thresh
    P = positive.sum() or 1

    order = np.argsort(-y_score)
    pos_sorted = positive[order]
    gains = np.cumsum(pos_sorted) / P

    deciles = np.arange(0, 101, 10)
    curve = []
    for d in deciles:
        k = int(round(d / 100 * n))
        curve.append(100 * (gains[k - 1] if k > 0 else 0.0))
    top20 = 100 * gains[int(round(0.2 * n)) - 1]
    return {"deciles": deciles.tolist(), "gain": curve, "top20_capture": float(top20)}


def bibliometric_baseline_score(df) -> np.ndarray:
    """Traditional proxy: sum of backward citations and claims. [paper]"""
    return (df["num_backward_refs"].astype(float).values +
            df["num_claims"].astype(float).values)
