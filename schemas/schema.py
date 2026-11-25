from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

GroundingStatus = Literal["supported", "partial", "insufficient_evidence"]


class SubQuestionItem(BaseModel):
    """A single straightforward sub-question derived from a complex essay prompt."""

    question_id: str = Field(description="Short identifier such as 'sq_1', 'sq_2'.")
    sub_question: str = Field(
        description="Focused, direct question or research angle to query against the literature corpus."
    )
    target_role: str = Field(
        default="evidence",
        description="Role in the essay, such as 'background_context', 'methods_comparison', 'empirical_results', or 'open_gap'.",
    )


class SubQuestionPlan(BaseModel):
    """Output of the splitter/planner agent for a complex prompt."""

    original_prompt: str = Field(description="The user's original complex prompt.")
    essay_goal: str = Field(
        description="Summary of the requested essay, background section, or research-question set."
    )
    sub_questions: List[SubQuestionItem] = Field(
        default_factory=list,
        description="Ordered list of focused sub-questions for retrieval.",
    )


class SourceCitation(BaseModel):
    """Exact provenance linking a claim or paragraph to a retrieved corpus chunk."""

    chunk_id: str = Field(description="Exact chunk_id returned by search_thesis_literature.")
    doc_id: str = Field(description="Document identifier (e.g., arXiv ID or filename stem).")
    source_file: str = Field(description="Exact PDF filename of the referenced paper.")
    page: str = Field(default="N/A", description="Page number or range from retrieved metadata.")
    section: str = Field(default="General", description="Detected section heading from metadata.")
    supporting_excerpt: str = Field(
        default="",
        description="Brief verbatim snippet or concise summary of the exact evidence used.",
    )


class EssayParagraphBlock(BaseModel):
    """A structured, evidence-grounded paragraph or argument block for essay synthesis."""

    heading: Optional[str] = Field(
        default=None,
        description="Optional sub-heading or thematic label for this block.",
    )
    grounding_status: GroundingStatus = Field(
        default="supported",
        description="Whether retrieved evidence fully supports, partially supports, or lacks evidence for this block.",
    )
    argument_text: str = Field(
        description="Synthesized academic prose grounded strictly in retrieved evidence without fabricating claims."
    )
    citations: List[SourceCitation] = Field(
        default_factory=list,
        description="Citations backing the claims in this block; empty if insufficient_evidence.",
    )
    caveat_or_gap: Optional[str] = Field(
        default=None,
        description="Explicit note on missing evidence, conflicting findings, or scope limits.",
    )


class EssaySynthesisOutput(BaseModel):
    """Structured output produced by the synthesizer agent from retrieved sub-question evidence."""

    essay_title: str = Field(description="Title or section heading for the synthesized essay.")
    main_prompt: str = Field(description="Original complex prompt being addressed.")
    answered_sub_questions: List[str] = Field(
        default_factory=list,
        description="Sub-questions addressed using retrieved corpus evidence.",
    )
    sections: List[EssayParagraphBlock] = Field(
        default_factory=list,
        description="Ordered structured blocks forming the essay or literature background.",
    )
    cross_paper_tensions_and_gaps: List[str] = Field(
        default_factory=list,
        description="Contrasts, limitations, or open gaps identified across retrieved papers.",
    )
    unanswered_or_low_evidence_topics: List[str] = Field(
        default_factory=list,
        description="Topics or sub-questions lacking sufficient evidence in the corpus.",
    )


class LatexCompilationOutput(BaseModel):
    """Final formatted LaTeX output produced by the LaTeX compiler/formatter agent."""

    format_profile: str = Field(
        default="thesis_chapter",
        description="Target formatting profile (e.g., 'thesis_chapter', 'acl_paper', 'ieee_conference', 'standalone_section').",
    )
    referencing_style: str = Field(
        default="author_year_natbib",
        description="Referencing convention used (e.g., 'author_year_natbib', 'numeric_biblatex', 'inline_doc_cite').",
    )
    latex_body: str = Field(
        description="Clean, compilable LaTeX section/essay content following the specified format and referencing style."
    )
    referenced_chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of verified chunk_ids referenced in the compiled LaTeX.",
    )
    formatting_notes: List[str] = Field(
        default_factory=list,
        description="Notes on macros used, citation keys, or items flagged as insufficient evidence.",
    )


# --- Backward-compatible aliases for existing imports ---
Citation = SourceCitation
ThesisSectionResponse = EssaySynthesisOutput