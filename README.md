# Predicting Commercialization Success of Renewable-Energy Patents

A hybrid multimodal framework combining **transformer text embeddings (BERT / ELECTRA)** with **gradient boosting (XGBoost / LightGBM)** to predict the commercialization success of renewable-energy patents (CPC class **Y02E**, USPTO, 2014–2023).

Commercialization success is measured as **assignment frequency** — the number of recorded ownership transfers — a forward-looking market signal, rather than backward-looking citation counts.

📄 **Paper:** *Predicting commercialization in renewable energy patents: A hybrid multimodal framework integrating BERT and ELECTRA embeddings with gradient boosting* — **World Patent Information** (Elsevier).
DOI: [10.1016/j.wpi.2026.102464](https://doi.org/10.1016/j.wpi.2026.102464)
   
💾 **Data:** [Zenodo record 21478629](https://zenodo.org/records/21478629)

---

## Key results

| Metric | Result |
|---|---|
| Coefficient of determination (R²) | **> 0.95** in all six datasets (max **0.962**) |
| Root mean squared error (RMSE) | **≈ 0.62** (best configuration) |
| Gain over structured-only baseline | **82.9% → 95.1%  (+12.2 points)** |
| Top predictor (mean \|SHAP\|) | **Assignee embeddings — 69.8%** |
| Semantic vs. structural contribution | **97.5% / 2.5%** (bootstrap *p* < 0.001) |
| Ranking stability | **Spearman ρ > 0.90** (1,000 resamples) |
| Temporal-leakage control | Patent age ↔ transfers correlation **≈ 0.01** |

**Main finding.** The organizational identity of the assignee dominates the technical content of the invention — empirical support for Teece's complementary-assets theory. As commercialization intensity rises (D3 → D8), the dominant driver shifts: **structural → organizational → fine-grained semantic**.

---

## Repository contents

| File | Role |
|---|---|
| `run_pipeline.py` | **Entry point** — runs the full pipeline end to end |
| `config.py` | Paths, thresholds, model and hyperparameter settings |
| `data.py` | Loading, cleaning and preparing the patent records |
| `embeddings.py` | BERT / ELECTRA encoding of the text fields (768-dim per field) |
| `modeling.py` | XGBoost / LightGBM training, tuning and cross-validation |
| `aggregation.py` | Aggregating results across folds, models and datasets |
| `statistics_tests.py` | Significance testing (bootstrap, Spearman rank stability) |
| `robustness.py` | Robustness checks, ablation and leakage control |
| `reporting.py` | Metric tables and figures |
| `requirements.txt` | Python dependencies |

---

## Data

The six threshold datasets are archived on Zenodo (not stored in this repository — the files are hundreds of megabytes):

**→ [https://zenodo.org/records/21478629](https://zenodo.org/records/21478629)**

`Data.zip` contains:

| File | Size | Threshold | Regime |
|---|---|---|---|
| `3input.json` | 329.7 MB | ≥ 3 transfers | broad market |
| `4input.json` | 186.6 MB | ≥ 4 transfers | |
| `5input.json` | 119.4 MB | ≥ 5 transfers | moderate |
| `6input.json` | 82.5 MB | ≥ 6 transfers | |
| `7input.json` | 57.3 MB | ≥ 7 transfers | |
| `8input.json` | 39.9 MB | ≥ 8 transfers | "innovation stars" |

Original sources: the public **USPTO Patent Assignment Dataset** and **Google Patents**, covering CPC class Y02E, 2014–2023.

### Setting up the data locally

1. Download `Data.zip` from the Zenodo link above.
2. Extract it into a `data/` folder next to the scripts:

```
.
├── run_pipeline.py
├── config.py
└── data/
    ├── 3input.json
    ├── 4input.json
    ├── 5input.json
    ├── 6input.json
    ├── 7input.json
    └── 8input.json
```

3. Make sure the data path in `config.py` points to this folder.

---

## Installation

```bash
git clone https://github.com/<your-username>/renewable-patent-commercialization.git
cd renewable-patent-commercialization

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python **3.10+** is recommended. A CUDA-capable GPU is strongly recommended for the embedding step (about 10 hours on a free Colab GPU); the gradient-boosting stage runs on a laptop CPU.

---

## Running

```bash
python run_pipeline.py
```

Settings — dataset paths, thresholds, models and hyperparameter ranges — are controlled from `config.py`.

---

## Method in brief

**Target variable.** Assignment frequency: the number of ownership transfers per patent. A transfer means a real market actor was willing to pay for the asset — a revealed preference, unlike citations.

**Datasets.** Six nested threshold datasets (D3–D8) treat commercialization as a spectrum rather than a binary outcome. This design revealed both the commercialization gradient and the performance-crossover phenomenon.

**Features.** Six text fields (abstract, claims, inventors, assignees, countries, citations) → 768 dimensions each = **4,608 semantic features**, concatenated with **5 structural indicators** (claim count, assignee count, country count, inventor count, backward citations).

**Models.** XGBoost and LightGBM, tuned with **Optuna (TPE)**, evaluated with **10-fold cross-validation**.

**Leakage control.** All time-derived variables (filing year, patent age) and all post-filing signals (forward citations) are excluded; only information available at filing time is used. Verified: patent age ↔ transfer count correlation ≈ 0.01.

**Interpretation.** SHAP (TreeExplainer) values aggregated per feature group, R²-weighted across 288 runs, with bootstrap significance testing and Spearman rank-stability checks.

### Ablation

| Configuration | R² | Δ vs. baseline |
|---|---|---|
| Structural only (LightGBM) | 82.9% | baseline |
| Text — BERT | 88.8% | +5.9 |
| Text — ELECTRA | 90.5% | +7.6 |
| Hybrid — BERT | 94.1% | +11.2 |
| Hybrid — ELECTRA | 95.1% | +12.2 |

No single-modality model suffices: semantic and structural signals are complementary.

---

## Citation

If you use this code or the datasets, please cite:

> Nazari, P., Pilevari, N., Shakeri, A., & Khadivar, A. Predicting commercialization in renewable energy patents: A hybrid multimodal framework integrating BERT and ELECTRA embeddings with gradient boosting. *World Patent Information*, Elsevier.

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).

---

## License

- **Code:** MIT License — see [`LICENSE`](LICENSE)
- **Data on Zenodo:** CC BY 4.0
- The published article is copyright of the publisher and is not redistributed here; it is linked by DOI above.

---

## Author

**Pouya Nazari** — PhD candidate, Technology Management
Islamic Azad University, Science and Research Branch, Tehran

Supervisor: Prof. Nazanin Pilevari · Advisors: Dr. Arnoosh Shakeri, Dr. Ameneh Khadivar
