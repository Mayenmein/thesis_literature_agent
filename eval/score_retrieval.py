import math
import re
from typing import Any, Dict, List, Set

from eval.retrieval_eval_schema import RetrievalOnlyEvalCase

# Weights for hierarchical provenance + exact chunk matching (sum = 1.0)
WEIGHT_DOC = 0.25
WEIGHT_PAGE = 0.20
WEIGHT_SECTION = 0.20
WEIGHT_CHUNK = 0.35

DEFAULT_HYBRID_PASS_THRESHOLD = 0.60

_CHUNK_ID_PATTERN = re.compile(r"^(?P<doc>.+?)_p(?P<page>\d+)_c(?P<chunk>\d+)$")


def _parse_chunk_id(chunk_id: str) -> tuple[str, str, str] | None:
    match = _CHUNK_ID_PATTERN.match((chunk_id or "").strip())
    if not match:
        return None
    return match.group("doc"), match.group("page"), match.group("chunk")


def _expected_doc_set(case: RetrievalOnlyEvalCase) -> Set[str]:
    docs: Set[str] = set()
    for d in case.expected_doc_ids:
        if d:
            docs.add(d.strip().lower())
    for sf in case.expected_source_files:
        if sf:
            stem = sf.strip().rsplit(".", 1)[0].lower()
            docs.add(stem)
            docs.add(sf.strip().lower())
    for cid in case.ground_truth_chunk_ids:
        parsed = _parse_chunk_id(cid)
        if parsed:
            docs.add(parsed[0].lower())
    return docs


def _expected_page_set(case: RetrievalOnlyEvalCase) -> Set[str]:
    pages: Set[str] = {p.strip() for p in case.expected_pages if str(p).strip()}
    for cid in case.ground_truth_chunk_ids:
        parsed = _parse_chunk_id(cid)
        if parsed:
            pages.add(parsed[1].lstrip("0") or "0")
    return pages


def _section_matches(actual_section: str, expected_sections: List[str]) -> bool:
    actual = (actual_section or "").strip().lower()
    if not actual or not expected_sections:
        return False
    for exp in expected_sections:
        exp_clean = (exp or "").strip().lower()
        if not exp_clean:
            continue
        if exp_clean in actual or actual in exp_clean:
            return True
        # Match by leading section number if present (e.g., "4.1" in "4.1 depth-aware rank schedule")
        num_match = re.match(r"^(\d+(?:\.\d+)*)", exp_clean)
        if num_match and re.search(rf"(?<!\d){re.escape(num_match.group(1))}(?!\d)", actual):
            return True
    return False


def compute_retrieval_metrics(
    case: RetrievalOnlyEvalCase,
    retrieved_chunk_ids: List[str],
    retrieved_metadata: List[Dict[str, Any]],
    hybrid_pass_threshold: float = DEFAULT_HYBRID_PASS_THRESHOLD,
) -> Dict[str, float]:
    k = case.thresholds.k
    top_k = retrieved_chunk_ids[:k]
    top_meta = retrieved_metadata[:k]
    gt: Set[str] = set(case.ground_truth_chunk_ids)

    if not case.is_answerable_in_corpus or len(gt) == 0:
        fp_count = len(top_k)
        abstention_score = 1.0 if fp_count == 0 else max(0.0, 1.0 - (fp_count / max(k, 1)))
        return {
            "recall_at_k": 1.0 if fp_count == 0 else 0.0,
            "precision_at_k": 1.0 if fp_count == 0 else 0.0,
            "mrr_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "doc_hit_rate": 1.0 if fp_count == 0 else 0.0,
            "false_positive_chunks": float(fp_count),
            "chunk_count_ratio": float(len(top_k)),
            "doc_match_rate": 1.0 if fp_count == 0 else 0.0,
            "page_match_rate": 1.0 if fp_count == 0 else 0.0,
            "section_match_rate": 1.0 if fp_count == 0 else 0.0,
            "hybrid_score": round(abstention_score, 4),
            "graded_passed": 1.0 if abstention_score >= hybrid_pass_threshold else 0.0,
        }

    # Standard binary exact-chunk IR metrics
    hits = [1 if cid in gt else 0 for cid in top_k]
    recall = sum(hits) / len(gt)
    precision = sum(hits) / max(len(top_k), 1)

    mrr = 0.0
    for rank, cid in enumerate(top_k, start=1):
        if cid in gt:
            mrr = 1.0 / rank
            break

    dcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(hits))
    ideal_hits = sorted(hits, reverse=True)
    idcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_hits))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    expected_docs = _expected_doc_set(case)
    expected_pages = _expected_page_set(case)
    expected_sections = case.expected_sections

    retrieved_files = {str(m.get("source_file", "")).strip().lower() for m in top_meta}
    expected_files = {f.strip().lower() for f in case.expected_source_files if f}
    doc_hit_rate = (
        len(retrieved_files & expected_files) / len(expected_files)
        if expected_files
        else 1.0
    )

    # Per-result graded relevance in [0.0, 1.0]
    graded_rels: List[float] = []
    doc_hits_per_item = 0
    page_hits_per_item = 0
    section_hits_per_item = 0
    matched_docs: Set[str] = set()
    matched_pages: Set[str] = set()
    matched_sections: Set[str] = set()

    for idx, cid in enumerate(top_k):
        meta = top_meta[idx] if idx < len(top_meta) and isinstance(top_meta[idx], dict) else {}
        parsed = _parse_chunk_id(cid)

        doc_val = str(meta.get("doc_id") or (parsed[0] if parsed else "")).strip().lower()
        src_val = str(meta.get("source_file") or "").strip().lower()
        src_stem = src_val.rsplit(".", 1)[0] if src_val else ""
        has_doc = bool(expected_docs) and (
            doc_val in expected_docs or src_val in expected_docs or src_stem in expected_docs
        )

        raw_page = str(meta.get("page") or (parsed[1] if parsed else "")).strip()
        norm_page = raw_page.lstrip("0") or ("0" if raw_page else "")
        has_page = bool(expected_pages) and (raw_page in expected_pages or norm_page in expected_pages)

        raw_section = str(meta.get("section") or "").strip()
        has_section = _section_matches(raw_section, expected_sections)

        is_exact_chunk = cid in gt
        rank_factor = 1.0 / math.log2(idx + 2)

        if has_doc:
            doc_hits_per_item += 1
            if doc_val:
                matched_docs.add(doc_val)
            if src_stem:
                matched_docs.add(src_stem)
        if has_page:
            page_hits_per_item += 1
            if norm_page:
                matched_pages.add(norm_page)
        if has_section:
            section_hits_per_item += 1
            matched_sections.add(raw_section.lower())

        chunk_component = (0.65 + 0.35 * rank_factor) if is_exact_chunk else 0.0
        item_score = (
            (WEIGHT_DOC if has_doc else 0.0)
            + (WEIGHT_PAGE if has_page else 0.0)
            + (WEIGHT_SECTION if has_section else 0.0)
            + WEIGHT_CHUNK * chunk_component
        )
        graded_rels.append(min(1.0, item_score))

    n_top = max(len(top_k), 1)
    doc_match_rate = doc_hits_per_item / n_top
    page_match_rate = page_hits_per_item / n_top
    section_match_rate = section_hits_per_item / n_top

    # Graded DCG / IDCG across top_k
    graded_dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(graded_rels))
    ideal_graded = sorted(graded_rels, reverse=True)
    graded_idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_graded))
    graded_ndcg = (graded_dcg / graded_idcg) if graded_idcg > 0 else 0.0

    # Coverage across expected dimensions so missing whole papers/pages/sections is reflected
    doc_cov = (len(matched_docs & expected_docs) / len(expected_docs)) if expected_docs else 1.0
    page_cov = (len(matched_pages & expected_pages) / len(expected_pages)) if expected_pages else 1.0
    sec_cov = (
        sum(1 for s in expected_sections if _section_matches(s, list(matched_sections)))
        / len(expected_sections)
        if expected_sections
        else 1.0
    )

    # Combine rank-weighted graded relevance (0.65) with provenance/recall coverage (0.35)
    mean_item_graded = sum(graded_rels) / n_top
    rank_quality = 0.5 * graded_ndcg + 0.5 * mean_item_graded
    coverage_quality = (
        0.30 * doc_cov
        + 0.20 * page_cov
        + 0.20 * sec_cov
        + 0.30 * recall
    )
    hybrid_score = 0.65 * rank_quality + 0.35 * coverage_quality
    ccr = len(top_k) / max(case.expected_chunk_count, 1)

    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "mrr_at_k": mrr,
        "ndcg_at_k": ndcg,
        "doc_hit_rate": doc_hit_rate,
        "false_positive_chunks": float(len(top_k) - sum(hits)),
        "chunk_count_ratio": ccr,
        "doc_match_rate": round(doc_match_rate, 4),
        "page_match_rate": round(page_match_rate, 4),
        "section_match_rate": round(section_match_rate, 4),
        "hybrid_score": round(hybrid_score, 4),
        "graded_passed": 1.0 if hybrid_score >= hybrid_pass_threshold else 0.0,
    }