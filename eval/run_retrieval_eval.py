import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from eval.retrieval_eval_schema import RetrievalOnlyEvalCase
from eval.score_retrieval import DEFAULT_HYBRID_PASS_THRESHOLD, compute_retrieval_metrics

RetrieverFn = Callable[[str, str], Tuple[List[str], List[Dict[str, Any]]]]


def evaluate_retrieval_suite(
    dataset_path: Path,
    retriever_fn: RetrieverFn,
    hybrid_pass_threshold: float = DEFAULT_HYBRID_PASS_THRESHOLD,
) -> Dict[str, Any]:
    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = [RetrievalOnlyEvalCase.model_validate(item) for item in raw_cases]

    per_case_results: List[Dict[str, Any]] = []
    strict_passed_count = 0
    graded_passed_count = 0

    for case in cases:
        chunk_ids, metadata = retriever_fn(
            case.retrieval_query,
            case.chunk_granularity_profile,
        )
        metrics = compute_retrieval_metrics(
            case,
            chunk_ids,
            metadata,
            hybrid_pass_threshold=hybrid_pass_threshold,
        )
        t = case.thresholds

        if case.is_answerable_in_corpus and len(case.ground_truth_chunk_ids) > 0:
            strict_passed = (
                metrics["recall_at_k"] >= t.min_recall_at_k
                and metrics["precision_at_k"] >= t.min_precision_at_k
                and metrics["mrr_at_k"] >= t.min_mrr_at_k
                and metrics["ndcg_at_k"] >= t.min_ndcg_at_k
                and metrics["doc_hit_rate"] >= t.min_doc_hit_rate
                and metrics["false_positive_chunks"] <= t.max_false_positive_chunks
            )
        else:
            strict_passed = metrics["false_positive_chunks"] <= t.max_false_positive_chunks

        graded_passed = bool(metrics["graded_passed"] >= 1.0)

        if strict_passed:
            strict_passed_count += 1
        if graded_passed:
            graded_passed_count += 1

        per_case_results.append(
            {
                "ret_eval_id": case.ret_eval_id,
                "passed": graded_passed,
                "graded_passed": graded_passed,
                "strict_passed": strict_passed,
                "hybrid_score": metrics["hybrid_score"],
                "metrics": metrics,
            }
        )

    total = len(per_case_results)
    denom = max(total, 1)
    summary = {
        "total_cases": total,
        "passed": graded_passed_count,
        "pass_rate": round(graded_passed_count / denom, 4),
        "strict_passed": strict_passed_count,
        "strict_pass_rate": round(strict_passed_count / denom, 4),
        "mean_hybrid_score": round(
            sum(r["metrics"]["hybrid_score"] for r in per_case_results) / denom, 4
        ),
        "mean_doc_match_rate": round(
            sum(r["metrics"]["doc_match_rate"] for r in per_case_results) / denom, 4
        ),
        "mean_page_match_rate": round(
            sum(r["metrics"]["page_match_rate"] for r in per_case_results) / denom, 4
        ),
        "mean_section_match_rate": round(
            sum(r["metrics"]["section_match_rate"] for r in per_case_results) / denom, 4
        ),
        "mean_recall_at_k": round(
            sum(r["metrics"]["recall_at_k"] for r in per_case_results) / denom, 4
        ),
        "mean_precision_at_k": round(
            sum(r["metrics"]["precision_at_k"] for r in per_case_results) / denom, 4
        ),
        "mean_mrr_at_k": round(
            sum(r["metrics"]["mrr_at_k"] for r in per_case_results) / denom, 4
        ),
        "mean_ndcg_at_k": round(
            sum(r["metrics"]["ndcg_at_k"] for r in per_case_results) / denom, 4
        ),
        "mean_doc_hit_rate": round(
            sum(r["metrics"]["doc_hit_rate"] for r in per_case_results) / denom, 4
        ),
        "results": per_case_results,
    }
    return summary