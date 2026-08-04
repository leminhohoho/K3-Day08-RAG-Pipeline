"""
Task 6 — Lexical Search Module (BM25 & TF-IDF).

BM25: rank_bm25.BM25Okapi(k1=1.5, b=0.75)
TF-IDF: sklearn char_wb ngram(3,5) — bonus 5 điểm

Corpus: load từ ChromaDB collection (Task 4) hoặc fallback load_documents + chunk_documents.
"""

import unicodedata
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .task4_chunking_indexing import load_documents, chunk_documents, get_collection


# =============================================================================
# CONFIGURATION
# =============================================================================

BM25_K1 = 1.5
BM25_B = 0.75
TOKEN_PATTERN = r"\w+(?:[-./]\w+)*"


# =============================================================================
# TEXT NORMALIZATION & TOKENIZATION
# =============================================================================

def normalize_lexical_text(text: str) -> str:
    """
    Normalize text for lexical search:
        - Unicode NFC
        - Lowercase
        - Collapse whitespace
        - Giữ số, năm, mã chương trình, dấu tiếng Việt
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = " ".join(text.split())
    return text


def tokenize_lexical(text: str) -> list[str]:
    """
    Unicode-aware tokenizer.
    Pattern: words with optional hyphens/dots/slashes (for codes like BUS1234, QH-2023-X).
    """
    return re.findall(TOKEN_PATTERN, text)


# =============================================================================
# CORPUS MANAGEMENT
# =============================================================================

@dataclass
class LexicalIndexState:
    corpus: list[dict]
    corpus_hash: str
    bm25: Optional[object] = None
    tfidf_vectorizer: Optional[object] = None
    tfidf_matrix: Optional[object] = None


_lexical_state: Optional[LexicalIndexState] = None


def _compute_corpus_hash(corpus: list[dict]) -> str:
    import hashlib
    h = hashlib.md5()
    for item in corpus:
        cid = item.get("metadata", {}).get("chunk_id", "")
        h.update(cid.encode())
        h.update(item.get("content", "").encode())
    return h.hexdigest()


def _ensure_empty_token_mapping(corpus: list[dict], tokenized: list[list[str]]) -> tuple[list[dict], list[list[str]]]:
    """Remove chunks that are empty after tokenization, fix index mapping."""
    filtered_corpus = []
    filtered_tokenized = []
    for item, tokens in zip(corpus, tokenized):
        if not tokens:
            continue
        filtered_corpus.append(item)
        filtered_tokenized.append(tokens)
    return filtered_corpus, filtered_tokenized


def load_lexical_corpus() -> list[dict]:
    """
    Load corpus: ưu tiên ChromaDB collection, fallback load_documents + chunk_documents.
    """
    # Try ChromaDB first
    try:
        collection = get_collection()
        if collection.count() > 0:
            all_data = collection.get(include=["documents", "metadatas"])
            corpus = []
            for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
                safe_meta = dict(meta) if meta else {}
                corpus.append({"content": doc, "metadata": safe_meta})
            if corpus:
                return corpus
    except Exception:
        pass

    # Fallback: load + chunk
    docs = load_documents()
    if not docs:
        return []
    return chunk_documents(docs)


def get_lexical_index_state(force_rebuild: bool = False) -> LexicalIndexState:
    """
    Lazy-load và cache lexical index state.
    Rebuild khi corpus hash đổi.
    """
    global _lexical_state

    corpus = load_lexical_corpus()
    if not corpus:
        _lexical_state = LexicalIndexState(corpus=[], corpus_hash="")
        return _lexical_state

    new_hash = _compute_corpus_hash(corpus)

    if _lexical_state is not None and not force_rebuild:
        if _lexical_state.corpus_hash == new_hash:
            return _lexical_state

    # Build new state
    state = LexicalIndexState(corpus=corpus, corpus_hash=new_hash)
    _lexical_state = state
    return state


# =============================================================================
# BM25 INDEX
# =============================================================================

def build_bm25_index(corpus: list[dict]):
    """
    Build BM25Okapi index từ corpus.
    Returns BM25Okapi instance hoặc None nếu corpus rỗng.
    """
    if not corpus:
        return None

    from rank_bm25 import BM25Okapi

    tokenized = [tokenize_lexical(normalize_lexical_text(item["content"])) for item in corpus]
    filtered_corpus, filtered_tokenized = _ensure_empty_token_mapping(corpus, tokenized)

    if not filtered_tokenized:
        return None

    return BM25Okapi(filtered_tokenized, k1=BM25_K1, b=BM25_B)


def _stable_lexical_sort_key(item: dict):
    """Sort key: (score desc, exact_match_count desc, document_id asc, chunk_index asc)."""
    meta = item.get("metadata") or {}
    content = item.get("content", "")
    return (
        -item["score"],
        -len([t for t in tokenize_lexical(content) if t in item.get("_query_tokens", [])]),
        meta.get("document_id", ""),
        meta.get("chunk_index", 0),
    )


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    BM25 lexical search.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,        # BM25 score
            'metadata': dict,
            'retrieval_method': 'bm25',
            'raw_scores': {'bm25': float, 'tfidf': None, 'rrf': None}
        }
    """
    normalized = normalize_lexical_text(query)
    if not normalized or top_k <= 0:
        return []

    state = get_lexical_index_state()
    if not state.corpus:
        return []

    if state.bm25 is None:
        state.bm25 = build_bm25_index(state.corpus)

    if state.bm25 is None:
        return []

    query_tokens = tokenize_lexical(normalized)
    if not query_tokens:
        return []

    scores = state.bm25.get_scores(query_tokens)

    # Get indices with positive scores
    scored_indices = [(i, float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
    # Sort by score descending
    scored_indices.sort(key=lambda x: (-x[1], x[0]))

    results = []
    for idx, score in scored_indices[:top_k]:
        item = state.corpus[idx]
        results.append({
            "content": item["content"],
            "score": score,
            "metadata": dict(item.get("metadata", {})),
            "retrieval_method": "bm25",
            "raw_scores": {"bm25": score, "tfidf": None, "rrf": None},
            "_query_tokens": query_tokens,
        })

    # Remove internal _query_tokens before returning
    for r in results:
        r.pop("_query_tokens", None)

    return results


# =============================================================================
# TF-IDF BONUS
# =============================================================================

def build_tfidf_index(corpus: list[dict]):
    """
    Build TF-IDF char n-gram index.

    Returns:
        (vectorizer, document_matrix) tuple hoặc (None, None) nếu corpus rỗng.
    """
    if not corpus:
        return None, None

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from scipy.sparse import issparse

    texts = [normalize_lexical_text(item["content"]) for item in corpus]
    if not any(texts):
        return None, None

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def _ensure_tfidf_ready(state: LexicalIndexState):
    """Ensure TF-IDF index is built in the state."""
    if state.tfidf_vectorizer is None and state.corpus:
        vec, mat = build_tfidf_index(state.corpus)
        state.tfidf_vectorizer = vec
        state.tfidf_matrix = mat


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    TF-IDF char n-gram search (bonus).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,        # cosine similarity [0, 1]
            'metadata': dict,
            'retrieval_method': 'tfidf',
            'raw_scores': {'bm25': None, 'tfidf': float, 'rrf': None}
        }
    """
    normalized = normalize_lexical_text(query)
    if not normalized or top_k <= 0:
        return []

    state = get_lexical_index_state()
    if not state.corpus:
        return []

    _ensure_tfidf_ready(state)

    if state.tfidf_vectorizer is None or state.tfidf_matrix is None:
        return []

    import numpy as np

    query_vec = state.tfidf_vectorizer.transform([normalized])
    scores = (state.tfidf_matrix @ query_vec.T).toarray().flatten()

    # Clamp [0, 1]
    scores = np.clip(scores, 0.0, 1.0)

    scored_indices = [(i, float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
    scored_indices.sort(key=lambda x: (-x[1], x[0]))

    results = []
    for idx, score in scored_indices[:top_k]:
        item = state.corpus[idx]
        results.append({
            "content": item["content"],
            "score": score,
            "metadata": dict(item.get("metadata", {})),
            "retrieval_method": "tfidf",
            "raw_scores": {"bm25": None, "tfidf": score, "rrf": None},
        })

    return results


# =============================================================================
# CONFIGURED LEXICAL SEARCH
# =============================================================================

def lexical_search_configured(
    query: str,
    top_k: int = 10,
    method: str = "bm25",
) -> list[dict]:
    """
    Configured lexical search: bm25 | tfidf | bm25_tfidf.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        method: "bm25", "tfidf", or "bm25_tfidf"

    Returns:
        List of results with appropriate retrieval_method.
    """
    if method == "bm25":
        return lexical_search(query, top_k=top_k)
    elif method == "tfidf":
        return tfidf_search(query, top_k=top_k)
    elif method == "bm25_tfidf":
        bm25_results = lexical_search(query, top_k=top_k * 2)
        tfidf_results = tfidf_search(query, top_k=top_k * 2)
        if not bm25_results and not tfidf_results:
            return []
        from .task7_reranking import rerank_rrf
        fused = rerank_rrf([bm25_results, tfidf_results], top_k=top_k)
        for item in fused:
            item["retrieval_method"] = "bm25_tfidf_rrf"
            if "raw_scores" not in item:
                item["raw_scores"] = {}
            item["raw_scores"]["rrf"] = item.get("score")
        return fused
    else:
        raise ValueError(f"Unknown lexical method: {method}")


def clear_lexical_cache():
    """Clear lexical index cache."""
    global _lexical_state
    _lexical_state = None


if __name__ == "__main__":
    # Test
    results = lexical_search("học phí thanh toán tín chỉ", top_k=5)
    print(f"BM25 results: {len(results)}")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['content'][:80]}...")

    print("\nTF-IDF results:")
    results2 = tfidf_search("học phí thanh toán tín chỉ", top_k=5)
    for r in results2:
        print(f"  [{r['score']:.3f}] {r['content'][:80]}...")