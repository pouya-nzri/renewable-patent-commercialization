"""
config.py
=========
Central configuration for the green-technology commercialization study.

Every experimental constant that appears in the paper's Methodology is defined
here once, so that the manuscript and the code cannot drift apart. Values that
map to a specific sentence or table in the paper are annotated with a [paper]
tag.

Design of the experimental space (paper, Section 3):
    2 encoders (BERT, ELECTRA)
  x 3 learners (LightGBM, XGBoost, CatBoost)
  x 6 datasets (D3 .. D8)
  x 10 folds
  = 360 model-dataset-fold runs                              [paper: "360 runs"]
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42                       # global seed for all RNGs
N_SPLITS: int = 10                          # [paper] 10-fold cross-validation
N_OPTUNA_TRIALS: int = 50                   # TPE trials per learner (raise for final run)

# --------------------------------------------------------------------------- #
# Datasets: the six nested assignment-frequency strata D3..D8
# Sample sizes are the population N reported in Table 3/4 of the paper and are
# used only for sanity-checking the loaded data, never to fabricate rows.
# --------------------------------------------------------------------------- #
STRATA: List[int] = [3, 4, 5, 6, 7, 8]                 # tau thresholds
EXPECTED_N: Dict[int, int] = {                          # [paper: Table 3]
    3: 33570, 4: 19150, 5: 12239, 6: 8519, 7: 5922, 8: 4168,
}

# --------------------------------------------------------------------------- #
# Text & structured feature definitions
# --------------------------------------------------------------------------- #
# Six textual fields, each embedded to a 768-dim CLS vector          [paper]
TEXT_FIELDS: List[str] = [
    "assignees",     # organizational identity  -> "Assignee Embeddings"
    "claims",        # legal/technical scope     -> "Claims Embeddings"
    "countries",     # jurisdictional footprint  -> "Country Embeddings"
    "backward_refs", # citation context          -> "Citation Embeddings"
    "inventors",     # inventor identity         -> "Inventor Embeddings"
    "abstract",      # disclosure semantics      -> "Abstract Embeddings"
]

# Five structured (count) indicators                                  [paper]
STRUCTURED_FIELDS: List[str] = [
    "num_claims",
    "num_assignees",
    "num_inventors",
    "num_countries",
    "num_backward_refs",
]

EMBED_DIM: int = 768
MAX_TOKEN_LEN: int = 512                     # [paper] transformer truncation length

# Human-readable group names used in every table/figure of the paper.
GROUP_DISPLAY: Dict[str, str] = {
    "assignees": "Assignee Embeddings",
    "claims": "Claims Embeddings",
    "countries": "Country Embeddings",
    "backward_refs": "Citation Embeddings",
    "inventors": "Inventor Embeddings",
    "abstract": "Abstract Embeddings",
    "num_claims": "Num. Claims",
    "num_assignees": "Num. Assignees",
    "num_inventors": "Num. Inventors",
    "num_countries": "Num. Countries",
    "num_backward_refs": "Num. Backward Refs",
}

# Modality map (for the semantic-vs-structural aggregation in the paper).
SEMANTIC_GROUPS: List[str] = list(TEXT_FIELDS)
STRUCTURAL_GROUPS: List[str] = list(STRUCTURED_FIELDS)

# --------------------------------------------------------------------------- #
# Encoders and learners
# --------------------------------------------------------------------------- #
ENCODERS: Dict[str, str] = {
    "bert": "bert-base-uncased",
    "electra": "google/electra-base-discriminator",  # discriminator = ELECTRA proper
}
LEARNERS: List[str] = ["lightgbm", "xgboost", "catboost"]

# Domain-specific encoders used only in the ablation ladder           [paper]
ABLATION_ENCODERS: Dict[str, str] = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "patentbert": "anferico/bert-for-patents",
}

# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
TARGET_COLUMN: str = "no_assignment"         # assignment frequency (regression target)
# NOTE: despite the field name, this is the *count of assignments* (the paper's
# "assignment frequency"), the continuous dependent variable.

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    embeddings_dir: str = "./artifacts/embeddings"
    models_dir: str = "./artifacts/models"
    results_dir: str = "./results"
    tables_dir: str = "./results/tables"
    figures_dir: str = "./results/figure_data"

    def make_all(self) -> None:
        for d in [self.data_dir, self.artifacts_dir, self.embeddings_dir,
                  self.models_dir, self.results_dir, self.tables_dir,
                  self.figures_dir]:
            os.makedirs(d, exist_ok=True)

    def data_file(self, tau: int) -> str:
        return os.path.join(self.data_dir, f"{tau}input.json")

    def embed_file(self, tau: int, encoder: str) -> str:
        return os.path.join(self.embeddings_dir, f"D{tau}_{encoder}.pkl")

    def model_file(self, tau: int, encoder: str, learner: str) -> str:
        return os.path.join(self.models_dir, f"D{tau}_{encoder}_{learner}_best.pkl")


PATHS = Paths()

# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def total_runs() -> int:
    """The 360 figure, computed rather than hard-coded."""
    return len(ENCODERS) * len(LEARNERS) * len(STRATA) * N_SPLITS


if __name__ == "__main__":
    print(f"Configured experimental space: {total_runs()} model-dataset-fold runs")
    assert total_runs() == 360, "Run count must equal 360 to match the paper."
    print("OK: matches the paper's 360 runs.")
