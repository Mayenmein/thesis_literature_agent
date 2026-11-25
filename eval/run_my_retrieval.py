import json
from pathlib import Path
from typing import Any

from eval.run_retrieval_eval import evaluate_retrieval_suite
from tools.tools import MAX_FINAL_CHUNKS, search_thesis_literature


def _extract_results(raw_response: Any) -> list[dict[str, Any]]:
    data = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
    if isinstance(data, dict):
        results = data.get("results", [])
        return [r for r in results if isinstance(r, dict)]
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def my_retriever_adapter(query: str, granularity_profile: str):
    max_results = MAX_FINAL_CHUNKS
    if granularity_profile and granularity_profile.startswith("top_"):
        suffix = granularity_profile.split("_", 1)[1]
        if suffix.isdigit():
            max_results = int(suffix)

    raw_response = search_thesis_literature(
        query_topic=query,
        max_results=max_results,
    )
    items = _extract_results(raw_response)

    chunk_ids = [str(item["chunk_id"]) for item in items if item.get("chunk_id")]
    metadata = [
        {
            "source_file": str(item.get("source_file", "")),
            "doc_id": str(item.get("doc_id", "")),
            "page": str(item.get("page", "")),
            "section": str(item.get("section", "")),
        }
        for item in items
    ]
    return chunk_ids, metadata


if __name__ == "__main__":
    print("Started the evaluation")
    report = evaluate_retrieval_suite(
        dataset_path=Path("eval/test_cases.json"),
        retriever_fn=my_retriever_adapter,
    )
    print(f"Graded Passed: {report['passed']}/{report['total_cases']} ({report['pass_rate']:.1%})")
    print(f"Strict Passed: {report['strict_passed']}/{report['total_cases']} ({report['strict_pass_rate']:.1%})")
    print(
        f"HybridScore: {report['mean_hybrid_score']:.4f} | "
        f"DocMatch: {report['mean_doc_match_rate']:.4f} | "
        f"PageMatch: {report['mean_page_match_rate']:.4f} | "
        f"SectionMatch: {report['mean_section_match_rate']:.4f}"
    )
    print(
        f"Recall@K: {report['mean_recall_at_k']:.4f} | "
        f"Precision@K: {report['mean_precision_at_k']:.4f} | "
        f"MRR@K: {report['mean_mrr_at_k']:.4f} | "
        f"nDCG@K: {report['mean_ndcg_at_k']:.4f} | "
        f"DocHit: {report['mean_doc_hit_rate']:.4f}"
    )

    # --- Diagnostic Breakdown on Sample Cases ---
    sample_ids = {
        "ret_eval_001",
        "ret_eval_003",
        "ret_eval_009",
        "ret_eval_011",
        "ret_eval_018",
        "ret_eval_025",
        "ret_eval_029",
        "ret_eval_050",
    }
    raw_cases = json.loads(Path("eval/test_cases.json").read_text(encoding="utf-8"))
    case_map = {c["ret_eval_id"]: c for c in raw_cases}

    print("\n--- Sample Case Breakdown ---")
    for rid in sample_ids:
        case_obj = case_map.get(rid)
        if not case_obj:
            continue
        r_ids, r_meta = my_retriever_adapter(
            case_obj["retrieval_query"],
            case_obj["chunk_granularity_profile"],
        )
        print(f"\n[{rid}] answerable={case_obj['is_answerable_in_corpus']}")
        print(f"  Expected chunks : {case_obj['ground_truth_chunk_ids']}")
        print(f"  Expected pages  : {case_obj['expected_pages']} | sections: {case_obj['expected_sections']}")
        print(f"  Retrieved chunks: {r_ids}")
        for idx, m in enumerate(r_meta[:5]):
            print(
                f"    #{idx + 1}: id={r_ids[idx] if idx < len(r_ids) else '?'} | "
                f"doc={m.get('doc_id')} | page={m.get('page')} | section={m.get('section')}"
            )