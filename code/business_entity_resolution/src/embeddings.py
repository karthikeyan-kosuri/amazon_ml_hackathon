"""
Multilingual sentence-embedding channel.

Why this exists: token/n-gram blocking (blocking.py) cannot match
"Lakshmi" against "లక్ష్మి" or handle French text the training data never
showed us — there's no shared substring or token in Latin/Telugu, and
France has zero training examples for any dictionary-based approach to
learn from. A general-purpose multilingual embedding model, by contrast,
was trained to place translation/transliteration-equivalent text nearby in
vector space regardless of script, so it generalizes to unseen languages
and scripts out of the box.

This is a general pretrained NLP model (Apache-2.0, ~118M params — well
under the 8B cap), not a lookup of this dataset's entities against an
external source, so it fits the "Allowed" list in the rules. It requires
one-time internet access to download weights from the model hub; after
that it runs fully offline.

This module degrades gracefully: if sentence-transformers or faiss aren't
installed, every function here returns None / empty and the rest of the
pipeline (blocking.py, features.py) already knows to skip this channel
via the `has_embedding_feature` flag — nothing else breaks.
"""

import numpy as np

from . import config

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

EMBEDDINGS_AVAILABLE = _ST_AVAILABLE and _FAISS_AVAILABLE

_model = None


def get_model():
    global _model
    if not _ST_AVAILABLE:
        return None
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 256) -> "np.ndarray | None":
    """Returns an (N, D) float32 L2-normalized embedding matrix, or None if
    the embedding stack isn't installed."""
    model = get_model()
    if model is None:
        return None
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # so inner product == cosine similarity
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_ann_index(embeddings: "np.ndarray"):
    """Flat inner-product index. For tens of millions of rows on a laptop,
    swap this for faiss.IndexIVFFlat with nlist~sqrt(n) trained on a
    sample — flagged in the README as the first thing to upgrade if
    embedding-blocking is too slow at full scale."""
    if not _FAISS_AVAILABLE:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def query_ann_index(index, query_embeddings: "np.ndarray", top_k: int):
    """Returns (scores, indices), each shape (num_queries, top_k)."""
    return index.search(query_embeddings, top_k)
