from typing import List, Literal
from pydantic import BaseModel, Field


class RetrievalMetricThresholds(BaseModel):
    k: int = Field(default=5, ge=1, le=20)
    min_recall_at_k: float = Field(ge=0.0, le=1.0)
    min_precision_at_k: float = Field(ge=0.0, le=1.0)
    min_mrr_at_k: float = Field(ge=0.0, le=1.0)
    min_ndcg_at_k: float = Field(ge=0.0, le=1.0)
    min_doc_hit_rate: float = Field(ge=0.0, le=1.0)
    max_false_positive_chunks: int = Field(default=0, ge=0)


class RetrievalOnlyEvalCase(BaseModel):
    ret_eval_id: str = Field(pattern=r"^ret_eval_\d{3}$")
    retrieval_query: str = Field(
        description="Direct query or sub-question sent to search_thesis_literature."
    )
    chunk_granularity_profile: Literal[
        "small_256",
        "medium_512",
        "large_1024",
        "parent_child_multi_chunk",
        "cross_chunk_hybrid",
    ]
    is_answerable_in_corpus: bool
    expected_chunk_count: int = Field(ge=0, le=15)
    ground_truth_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Exact chunk_ids in papers_db that ground the answer (empty if unanswerable).",
    )
    expected_doc_ids: List[str] = Field(default_factory=list)
    expected_source_files: List[str] = Field(default_factory=list)
    expected_sections: List[str] = Field(default_factory=list)
    expected_pages: List[str] = Field(default_factory=list)
    bm25_anchor_terms: List[str] = Field(default_factory=list)
    thresholds: RetrievalMetricThresholds