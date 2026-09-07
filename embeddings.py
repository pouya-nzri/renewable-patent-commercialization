"""
embeddings.py
=============
Frozen transformer embeddings for the six textual fields.

Implements the paper's "frozen semantic representation" leakage control:
    * models are loaded in inference mode (eval + no_grad),
    * weights are NEVER fine-tuned on the outcome,
    * each field -> 768-dim [CLS] vector.                              [paper]

Embeddings are cached to disk per (stratum, encoder) so the expensive
forward passes run only once.
"""

from __future__ import annotations
import os
from typing import Dict, List

import numpy as np
import joblib

import config as C

# Torch/transformers are imported lazily so that the statistics and
# table-generation modules can be used without a GPU stack installed.


def _load_encoder(encoder_key: str):
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = C.ENCODERS.get(encoder_key) or C.ABLATION_ENCODERS.get(encoder_key)
    if model_name is None:
        raise ValueError(f"Unknown encoder '{encoder_key}'.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()                      # <- frozen, inference mode
    return tokenizer, model, device


def embed_texts(texts: List[str], encoder_key: str,
                batch_size: int = 32) -> np.ndarray:
    """
    Return an (n, 768) array of [CLS] embeddings for `texts`.

    Batched for throughput; deterministic (no dropout in eval mode).
    """
    import torch

    tokenizer, model, device = _load_encoder(encoder_key)
    out = np.empty((len(texts), C.EMBED_DIM), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", max_length=C.MAX_TOKEN_LEN,
                truncation=True, padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            cls = model(**enc).last_hidden_state[:, 0, :]   # [CLS] token
            out[start:start + len(batch)] = cls.cpu().numpy()
    return out


def build_field_embeddings(df, encoder_key: str,
                           tau: int, cache: bool = True) -> Dict[str, np.ndarray]:
    """
    Produce a dict {field: (n,768)} for all six textual fields under one
    encoder, caching the whole dict per (stratum, encoder).
    """
    cache_path = C.PATHS.embed_file(tau, encoder_key)
    if cache and os.path.exists(cache_path):
        return joblib.load(cache_path)

    emb: Dict[str, np.ndarray] = {}
    for field in C.TEXT_FIELDS:
        col = f"{field}_text"
        print(f"  [embed] D{tau} {encoder_key}:{field} ...")
        emb[field] = embed_texts(df[col].tolist(), encoder_key)

    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump(emb, cache_path)
    return emb


def assemble_matrix(struct: np.ndarray,
                    field_emb: Dict[str, np.ndarray]) -> "tuple[np.ndarray, list, dict]":
    """
    Concatenate [structured | field embeddings] into one design matrix and
    return (X, feature_names, group_slices).

    group_slices maps each group name (structured field or text field) to the
    (start, end) column span it occupies. This is what makes *grouped* SHAP
    possible downstream: SHAP mass is summed within each 768-dim block. [paper]
    """
    blocks = [struct]
    names: List[str] = list(C.STRUCTURED_FIELDS)
    slices: Dict[str, tuple] = {}

    col = 0
    for f in C.STRUCTURED_FIELDS:                      # structured first
        slices[f] = (col, col + 1)
        col += 1

    for field in C.TEXT_FIELDS:                        # then each embedding block
        block = field_emb[field]
        blocks.append(block)
        start = col
        names.extend([f"{field}_emb_{i}" for i in range(block.shape[1])])
        col += block.shape[1]
        slices[field] = (start, col)

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    return X, names, slices
