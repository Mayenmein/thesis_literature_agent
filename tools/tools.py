import os
import hashlib
import math
import re
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

# --- Configuration ---
DB_PATH = "./papers_db"
COLLECTION_NAME = "thesis_literature_corpus"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Configuration ---
DB_PATH = "./papers_db"
COLLECTION_NAME = "thesis_literature_corpus"

# Retrieval & gating settings (tuned for multi-question essay synthesis)
CANDIDATE_POOL_SIZE = 20        # Pull top N from dense and BM25 before RRF fusion
MAX_FINAL_CHUNKS = 8            # Default max chunks returned per sub-query
DEFAULT_MAX_PER_DOC = 2         # Prevents a single chunk/paper from crowding out diversity
RRF_K = 60                      # Standard Reciprocal Rank Fusion constant

# Confidence thresholds (tune against your corpus and embedding/reranker scores)
MIN_RRF_SCORE = 0.008           # Minimum RRF score when reranker is disabled
USE_RERANKER = False            # Set True after `pip install sentence-transformers`
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MIN_RERANK_SCORE = -2.0         # Logit threshold for CrossEncoder

# Preserve original casing in text/metadata; tokenize on Unicode word tokens for BM25
WORD_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)

_bm25_index: BM25Okapi | None = None
_corpus_ids: list[str] = []
_corpus_docs: list[str] = []
_corpus_metas: list[dict[str, Any]] = []
_corpus_signature: str | None = None
_reranker: Any | None = None


def tokenize_words(text: str) -> list[str]:
    """Splits text into word tokens while preserving original letter casing."""
    if not text:
        return []
    return WORD_TOKEN_PATTERN.findall(text)


def _get_reranker() -> Any | None:
    """Lazily loads the cross-encoder reranker only when enabled."""
    global _reranker
    if not USE_RERANKER:
        return None
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def _compute_corpus_signature(ids: list[str], metas: list[dict[str, Any]]) -> str:
    """Computes a lightweight signature from chunk IDs and ingestion timestamps."""
    hasher = hashlib.md5()
    for idx, cid in enumerate(ids):
        meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
        ingested = str(meta.get("ingested_at", ""))
        hasher.update(f"{cid}:{ingested}|".encode("utf-8", errors="replace"))
    return hasher.hexdigest()


def _refresh_bm25_index(force: bool = False) -> bool:
    """Loads documents from ChromaDB and builds an in-memory BM25Okapi word-token index."""
    global _bm25_index, _corpus_ids, _corpus_docs, _corpus_metas, _corpus_signature

    current_count = collection.count()
    if current_count == 0:
        _bm25_index = None
        _corpus_ids = []
        _corpus_docs = []
        _corpus_metas = []
        _corpus_signature = None
        return False

    stored = collection.get(include=["documents", "metadatas"])
    ids = stored.get("ids") or []
    docs = stored.get("documents") or []
    metas = stored.get("metadatas") or []

    new_sig = _compute_corpus_signature(ids, metas)
    if not force and _bm25_index is not None and new_sig == _corpus_signature:
        return True

    valid_ids: list[str] = []
    valid_docs: list[str] = []
    valid_metas: list[dict[str, Any]] = []
    tokenized_corpus: list[list[str]] = []

    for idx, doc in enumerate(docs):
        if not doc or not doc.strip():
            continue
        tokens = tokenize_words(doc)
        if not tokens:
            continue
        doc_id = ids[idx] if idx < len(ids) else f"doc_{idx}"
        meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
        valid_ids.append(str(doc_id))
        valid_docs.append(doc)
        valid_metas.append(meta)
        tokenized_corpus.append(tokens)

    if not tokenized_corpus:
        _bm25_index = None
        _corpus_ids = []
        _corpus_docs = []
        _corpus_metas = []
        _corpus_signature = new_sig
        return False

    _corpus_ids = valid_ids
    _corpus_docs = valid_docs
    _corpus_metas = valid_metas
    _bm25_index = BM25Okapi(tokenized_corpus)
    _corpus_signature = new_sig
    return True


def _dense_search(query_topic: str, top_k: int) -> list[dict[str, Any]]:
    """Queries ChromaDB for dense semantic candidates."""
    total_docs = collection.count()
    if total_docs == 0:
        return []

    results = collection.query(
        query_texts=[query_topic],
        n_results=min(top_k, total_docs),
        include=["documents", "metadatas", "distances"],
    )

    ids = (results.get("ids") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    candidates: list[dict[str, Any]] = []
    for rank, doc in enumerate(docs, start=1):
        if not doc:
            continue
        chunk_id = str(ids[rank - 1]) if rank - 1 < len(ids) else f"dense_{rank}"
        meta = metas[rank - 1] if rank - 1 < len(metas) and isinstance(metas[rank - 1], dict) else {}
        dist = distances[rank - 1] if rank - 1 < len(distances) else None
        candidates.append({
            "id": chunk_id,
            "document": doc,
            "metadata": meta,
            "dense_rank": rank,
            "dense_distance": dist,
        })
    return candidates


def _bm25_search(query_topic: str, top_k: int) -> list[dict[str, Any]]:
    """Scores corpus chunks using word-token BM25 and returns top candidates."""
    if _bm25_index is None or not _corpus_docs:
        return []

    query_tokens = tokenize_words(query_topic)
    if not query_tokens:
        return []

    scores = _bm25_index.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: float(scores[i]),
        reverse=True,
    )[:top_k]

    candidates: list[dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        score = float(scores[idx])
        if score <= 0.0:
            continue
        candidates.append({
            "id": _corpus_ids[idx],
            "document": _corpus_docs[idx],
            "metadata": _corpus_metas[idx],
            "bm25_rank": rank,
            "bm25_score": score,
        })
    return candidates


def _reciprocal_rank_fusion(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuses dense and BM25 rankings using Reciprocal Rank Fusion."""
    fused: dict[str, dict[str, Any]] = {}

    def _upsert(item: dict[str, Any], rank: int, source_key: str) -> None:
        chunk_id = item["id"]
        rrf_contrib = 1.0 / (rrf_k + rank)
        if chunk_id not in fused:
            fused[chunk_id] = {
                "id": chunk_id,
                "document": item["document"],
                "metadata": item["metadata"],
                "rrf_score": 0.0,
                "dense_rank": None,
                "dense_distance": None,
                "bm25_rank": None,
                "bm25_score": None,
            }
        fused[chunk_id]["rrf_score"] += rrf_contrib
        if source_key == "dense":
            fused[chunk_id]["dense_rank"] = rank
            fused[chunk_id]["dense_distance"] = item.get("dense_distance")
        elif source_key == "bm25":
            fused[chunk_id]["bm25_rank"] = rank
            fused[chunk_id]["bm25_score"] = item.get("bm25_score")

    for hit in dense_hits:
        _upsert(hit, hit["dense_rank"], "dense")

    for hit in bm25_hits:
        _upsert(hit, hit["bm25_rank"], "bm25")

    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


def _apply_doc_cap(
    candidates: list[dict[str, Any]],
    max_results: int,
    max_per_doc: int | None,
) -> list[dict[str, Any]]:
    """Selects up to max_results while optionally capping chunks per source document."""
    selected: list[dict[str, Any]] = []
    doc_counts: dict[str, int] = {}

    for item in candidates:
        meta = item.get("metadata") or {}
        doc_key = str(meta.get("doc_id") or meta.get("source_file") or item["id"])
        if max_per_doc is not None and max_per_doc > 0:
            if doc_counts.get(doc_key, 0) >= max_per_doc:
                continue
        selected.append(item)
        doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
        if len(selected) >= max_results:
            break

    return selected


def search_thesis_literature(
    query_topic: str,
    max_results: int = MAX_FINAL_CHUNKS,
    max_per_doc: int | None = DEFAULT_MAX_PER_DOC,
) -> dict[str, Any]:
    """Searches ingested research papers using hybrid retrieval (BM25 + dense embeddings + RRF),

    optional cross-encoder reranking, and confidence gating so low-relevance chunks
    are excluded before reaching the agent.

    Args:
        query_topic: A focused sub-question, keyword, methodology, model architecture,
            dataset detail, or research claim to verify against the corpus.
        max_results: Maximum number of evidence chunks to return for this query.
        max_per_doc: Optional cap on chunks returned from the same paper per query.
    """
    try:
        cleaned_query = (query_topic or "").strip()
        if not cleaned_query:
            return {
                "status": "empty",
                "query": query_topic,
                "confident_match": False,
                "count": 0,
                "results": [],
                "message": "Query topic was empty; no chunks retrieved.",
            }

        has_docs = _refresh_bm25_index()
        if not has_docs:
            return {
                "status": "empty",
                "query": cleaned_query,
                "confident_match": False,
                "count": 0,
                "results": [],
                "message": "No ingested literature chunks available in the database.",
            }

        effective_limit = max(1, int(max_results))
        pool_size = max(CANDIDATE_POOL_SIZE, effective_limit * 3)

        dense_hits = _dense_search(cleaned_query, top_k=pool_size)
        bm25_hits = _bm25_search(cleaned_query, top_k=pool_size)
        fused_candidates = _reciprocal_rank_fusion(dense_hits, bm25_hits, rrf_k=RRF_K)

        if not fused_candidates:
            return {
                "status": "no_confident_match",
                "query": cleaned_query,
                "confident_match": False,
                "count": 0,
                "results": [],
                "message": "No matching passages found in the corpus.",
            }

        top_pool = fused_candidates[:pool_size]
        reranker = _get_reranker()

        accepted: list[dict[str, Any]] = []
        if reranker is not None:
            pairs = [(cleaned_query, c["document"]) for c in top_pool]
            raw_scores = reranker.predict(pairs)
            for cand, raw_score in zip(top_pool, raw_scores):
                score_val = float(raw_score)
                if math.isnan(score_val) or score_val < MIN_RERANK_SCORE:
                    continue
                updated = dict(cand)
                updated["rerank_score"] = round(score_val, 4)
                accepted.append(updated)
            accepted.sort(key=lambda x: x["rerank_score"], reverse=True)
        else:
            for cand in top_pool:
                if float(cand["rrf_score"]) < MIN_RRF_SCORE:
                    continue
                accepted.append(cand)

        final_candidates = _apply_doc_cap(
            accepted,
            max_results=effective_limit,
            max_per_doc=max_per_doc,
        )

        if not final_candidates:
            return {
                "status": "no_confident_match",
                "query": cleaned_query,
                "confident_match": False,
                "count": 0,
                "results": [],
                "message": (
                    "Retrieved passages did not meet the minimum relevance threshold; "
                    "returning no chunks to avoid low-confidence context."
                ),
            }

        formatted_results: list[dict[str, Any]] = []
        for item in final_candidates:
            meta = item.get("metadata") or {}
            source_file = meta.get("source_file", "Unknown Paper")
            doc_id = meta.get("doc_id") or (
                str(source_file).rsplit(".", 1)[0] if source_file else "unknown_doc"
            )
            entry: dict[str, Any] = {
                "chunk_id": item["id"],
                "doc_id": doc_id,
                "source_file": source_file,
                "page": meta.get("page", "N/A"),
                "section": meta.get("section", "General"),
                "chunk_index": meta.get("chunk_index", "N/A"),
                "rrf_score": round(float(item["rrf_score"]), 6),
                "content": item["document"].strip(),
            }
            if item.get("dense_rank") is not None:
                entry["dense_rank"] = item["dense_rank"]
            if item.get("bm25_rank") is not None:
                entry["bm25_rank"] = item["bm25_rank"]
            if "rerank_score" in item:
                entry["rerank_score"] = item["rerank_score"]
            if item.get("bm25_score") is not None:
                entry["bm25_score"] = round(float(item["bm25_score"]), 4)
            formatted_results.append(entry)

        return {
            "status": "success",
            "query": cleaned_query,
            "confident_match": True,
            "count": len(formatted_results),
            "results": formatted_results,
        }

    except Exception as e:
        return {
            "status": "error",
            "query": query_topic,
            "confident_match": False,
            "count": 0,
            "results": [],
            "message": f"Error querying literature database: {str(e)}",
        }


def list_ingested_papers() -> dict[str, Any]:
    """Lists all unique research paper files currently ingested and available in the local ChromaDB literature corpus.

    Takes no arguments.
    """
    try:
        all_data = collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas") or []

        if not metadatas:
            return {
                "status": "empty",
                "total_count": 0,
                "papers": [],
                "message": (
                    "The literature database is currently empty. No papers have been"
                    " ingested yet."
                ),
            }

        unique_papers = sorted(
            list({
                meta.get("source_file", "Unknown")
                for meta in metadatas
                if isinstance(meta, dict)
            })
        )

        return {
            "status": "success",
            "total_count": len(unique_papers),
            "papers": unique_papers,
        }

    except Exception as e:
        return {
            "status": "error",
            "total_count": 0,
            "papers": [],
            "message": f"Error retrieving paper list: {str(e)}",
        }