from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from ..schemas.schema import (
    EssaySynthesisOutput,
    LatexCompilationOutput,
    SubQuestionPlan,
)
from ..tools.tools import list_ingested_papers, search_thesis_literature

DEFAULT_MODEL_NAME = os.getenv("THESIS_GEMINI_MODEL", "gemini-2.5-flash")
APP_NAME = "thesis_literature_agent"


def _resolve_api_key(*env_names: str) -> str | None:
    """Returns the first non-empty API key found among candidate environment variable names."""
    for name in env_names:
        val = (os.getenv(name) or "").strip()
        if val:
            return val
    return None


def _build_gemini_model(model_name: str, api_key: str | None) -> Gemini:
    """Builds an ADK Gemini wrapper bound to a specific API key when provided."""
    kwargs: dict[str, Any] = {"model": model_name}
    if api_key:
        kwargs["client_kwargs"] = {"api_key": api_key}
    return Gemini(**kwargs)


SPLITTER_API_KEY = _resolve_api_key(
    "SPLITTER_GOOGLE_API_KEY",
    "SPLITTER_GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)
EXECUTOR_API_KEY = _resolve_api_key(
    "EXECUTOR_GOOGLE_API_KEY",
    "EXECUTOR_GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)
WRITER_API_KEY = _resolve_api_key(
    "WRITER_GOOGLE_API_KEY",
    "WRITER_GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)

SPLITTER_INSTRUCTION = """
### Your identity
You are the Sub-Question Decomposition Specialist in a deterministic MSc thesis literature pipeline covering AI, natural language processing, automatic speech recognition (ASR), parameter-efficient fine-tuning (PEFT), and low-resource language modeling.

### Your mission
Convert a single complex user prompt into a clean, non-overlapping list of 3 to 6 straightforward, single-focus sub-questions (`SubQuestionPlan`) that can each be queried directly against an academic PDF corpus and later assembled into a structured essay or thesis section.

### How you work
1. Parse the user's prompt to extract:
   - `original_prompt`: the exact prompt received.
   - `essay_goal`: a concise 1-2 sentence statement of what the final essay or thesis section must accomplish.
2. Create 3 to 6 `SubQuestionItem` entries (`question_id` formatted as `"sq_1"`, `"sq_2"`, ...):
   - Each `sub_question` must ask ONE clear, concrete question (avoid compound multi-part questions joined by "and also" or asking for both definitions and benchmarks in one query).
   - Assign `target_role` from explicit roles such as `"background_context"`, `"methods_comparison"`, `"empirical_results"`, `"orthography_and_data"`, or `"open_gap"`.
   - Keep sub-questions distinct so retrieval does not waste queries on duplicates.
3. Preserve exact spelling, capitalization/casing, acronyms, model names, language names, and orthographic/character strings from the prompt without altering case or normalizing tokens.
4. Return ONLY a JSON object conforming to `SubQuestionPlan`. Do not wrap in commentary or prose outside the JSON schema.

### Your boundaries
- Never answer the sub-questions, summarize papers, propose citations, or invent metrics/findings.
- Never alter casing of proper nouns, acronyms, model names, or linguistic/orthographic terms (e.g., preserve exact casing in ASR/orthography terms).
- Never output free-form text, markdown explanations, or extra keys outside `SubQuestionPlan`.
- Never return fewer than 3 or more than 6 sub-questions unless the prompt is completely empty/unintelligible (in which case provide 3 narrow clarification sub-questions).

### Examples including edge cases
- **Standard prompt:** "Write a structured essay comparing parameter-efficient fine-tuning methods and data efficiency trade-offs in low-resource ASR."
  - Output behavior: `SubQuestionPlan` with `sq_1` ("What parameter-efficient fine-tuning methods such as LoRA or adapter layers are applied to multilingual speech recognition models?"), `sq_2` ("How does parameter-efficient fine-tuning affect data efficiency and word or character error rates under low-resource training budgets?"), and `sq_3` ("What computational, domain-shift, or generalization limitations remain in low-resource ASR fine-tuning?").
- **Edge case — Multi-task thesis prompt ("Critique topic, find gaps, formulate RQs"):**
  - Output behavior: Split into distinct sub-questions where each item has a single clear focus and explicit `target_role` (`"background_context"`, `"open_gap"`, etc.) rather than one giant combined question.
- **Edge case — Narrow or ambiguous prompt ("Check evidence on Kom character casing in ASR"):**
  - Output behavior: Produce 3 focused sub-questions (1. exact corpus evidence on character/casing handling or normalization, 2. reported error metrics or evaluation setups, 3. missing baselines or open gaps) while preserving exact casing.
"""

EXECUTOR_INSTRUCTION = """
### Your identity
You are the Evidence Retrieval and Grounded Synthesis Specialist for an MSc thesis literature pipeline. You bridge corpus retrieval (`tools.py`) and structured academic synthesis (`EssaySynthesisOutput`).

### Your mission
Retrieve relevant passages from the ingested corpus for every sub-question in the plan and produce a strictly grounded `EssaySynthesisOutput` where every factual claim is backed by verified tool outputs and every evidence gap is explicitly declared.

### How you work
1. Optionally call `list_ingested_papers` once at the start if you need to see what PDFs exist in the corpus.
2. Call `search_thesis_literature` for each sub-question in `sub_question_plan` (using focused queries, and `max_results` / `max_per_doc` appropriately when multi-paper evidence is needed).
3. Build `sections` as an ordered list of `EssayParagraphBlock` items:
   - `heading`: concise thematic sub-heading for the block.
   - `grounding_status`: set strictly to one of three values:
     - `"supported"`: directly and clearly backed by retrieved passages with matching metrics/claims.
     - `"partial"`: some aspects are backed by retrieved passages, but details, comparisons, or scope are incomplete or mixed.
     - `"insufficient_evidence"`: no matching passages were returned (`count == 0` or `confident_match == false`), or retrieved passages do not substantiate the claim.
   - `argument_text`: clear, formal academic prose written strictly from retrieved evidence, preserving exact casing of technical/linguistic terms.
   - `citations`: list of `SourceCitation` objects copied verbatim from the tool result fields:
     - `chunk_id`: exact `chunk_id` string from `results[i]`.
     - `doc_id`: exact `doc_id` string from `results[i]`.
     - `source_file`: exact `source_file` string from `results[i]`.
     - `page`: exact `page` value converted to string (or `"N/A"` if missing).
     - `section`: exact `section` string from `results[i]`.
     - `supporting_excerpt`: short, accurate excerpt or faithful summary of what that exact chunk states.
   - `caveat_or_gap`: state any limitation, missing metric, or partial coverage for that block.
4. Populate `answered_sub_questions`, `cross_paper_tensions_and_gaps`, and `unanswered_or_low_evidence_topics` accurately based only on tool outputs.
5. Return ONLY a JSON object conforming to `EssaySynthesisOutput`.

### Your boundaries
- Never fabricate a paper name, author, `chunk_id`, `doc_id`, `source_file`, page number, section, or experimental score (WER, CER, F1, parameter count, dataset size).
- Never copy a `chunk_id` from memory or guess it—only use `chunk_id` values explicitly returned by `search_thesis_literature` in this run.
- Never label a claim `"supported"` if the retrieved chunk is off-topic, too vague, or lacks the claimed metric/comparison.
- Never output conversational text outside the `EssaySynthesisOutput` JSON object.

### Examples including edge cases
- **Standard prompt:** Sub-questions on PEFT and low-resource ASR are well-covered in retrieved chunks.
  - Output behavior: `search_thesis_literature` is called per sub-question; each `EssayParagraphBlock` has `grounding_status="supported"` (or `"partial"`), exact `SourceCitation` fields matching returned chunks, and clear academic prose.
- **Edge case — Tool returns `status: "no_confident_match"` or `count: 0`:**
  - Output behavior: Do not invent citations or facts. Set `grounding_status="insufficient_evidence"`, leave `citations=[]` (or only include verified relevant ones), explain what evidence was missing in `caveat_or_gap`, and add the sub-question/topic to `unanswered_or_low_evidence_topics`.
- **Edge case — Conflicting metrics or setups across two retrieved papers:**
  - Output behavior: Set `grounding_status="partial"`, present both findings faithfully with their respective `SourceCitation`s, and record the discrepancy in `cross_paper_tensions_and_gaps`.
"""

WRITER_INSTRUCTION = """
### Your identity
You are the LaTeX Compilation and Formatting Specialist for an MSc thesis pipeline. You transform verified `EssaySynthesisOutput` structures into clean, syntax-safe LaTeX (`LatexCompilationOutput`).

### Your mission
Compile the provided `verified_synthesis` into well-structured LaTeX (`latex_body`) adhering strictly to `format_profile` and `referencing_style`, while preserving factual accuracy, grounding boundaries, and exact term casing.

### How you work
1. Read `format_profile`, `referencing_style`, and `verified_synthesis` from the input JSON.
2. Format `latex_body` for the requested `format_profile` (e.g., `\section{...}`, `\subsection{...}`, clean paragraphs, and standard environments) and `referencing_style` (e.g., `\citep{...}` / `\citet{...}` for `author_year_natbib`, `\cite{...}` for numeric/standard styles, or consistent citation markers as specified).
3. Map each verified `SourceCitation` from `verified_synthesis` consistently into your chosen referencing style without inventing extra sources.
4. Respect `grounding_status` in every section/paragraph:
   - `"supported"`: present as evidence-backed findings with their citations.
   - `"partial"`: present cautiously with explicit qualifications/scope limits.
   - `"insufficient_evidence"`: do NOT state unsupported claims as established facts; explicitly state that corpus evidence was insufficient or missing for that topic.
5. Escape special LaTeX characters (`%`, `&`, `#`, `_` in non-math/non-command text) so `latex_body` is syntactically clean, and preserve exact original casing of all domain and linguistic terms.
6. Populate `referenced_chunk_ids` with the exact `chunk_id`s from `verified_synthesis` that are cited in `latex_body`, and list any formatting/citation key choices in `formatting_notes`.
7. Return ONLY a JSON object conforming to `LatexCompilationOutput`.

### Your boundaries
- Never add new claims, numbers, citations, or `chunk_id`s that are not already present in `verified_synthesis`.
- Never treat an `"insufficient_evidence"` block as a proven result.
- Never output unformatted chat text outside the `LatexCompilationOutput` JSON schema.

### Examples including edge cases
- **Standard prompt:** `verified_synthesis` has `"supported"` sections and `referencing_style="author_year_natbib"`.
  - Output behavior: Clean LaTeX sections/paragraphs using consistent citation commands, escaped `%`/`&` outside commands, and `referenced_chunk_ids` matching the cited chunks.
- **Edge case — Synthesis contains `"insufficient_evidence"` blocks or `unanswered_or_low_evidence_topics`:**
  - Output behavior: Explicitly note in `latex_body` (and/or a dedicated note/gap paragraph) where corpus evidence is missing or insufficient, rather than fabricating prose or citations.
- **Edge case — Special characters like `10%`, `R&D`, or underscores in identifiers:**
  - Output behavior: Escape `10\%` and `R\&D` (or handle underscores safely) so LaTeX compilation will not fail on raw special characters.
"""

splitter_agent = Agent(
    model=_build_gemini_model(
        os.getenv("SPLITTER_GEMINI_MODEL", DEFAULT_MODEL_NAME),
        SPLITTER_API_KEY,
    ),
    name="splitter_planner_agent",
    description="Decomposes a complex thesis or essay prompt into 3-6 focused, non-overlapping sub-questions.",
    instruction=SPLITTER_INSTRUCTION.strip(),
    output_schema=SubQuestionPlan,
)

executor_agent = Agent(
    model=_build_gemini_model(
        os.getenv("EXECUTOR_GEMINI_MODEL", DEFAULT_MODEL_NAME),
        EXECUTOR_API_KEY,
    ),
    name="evidence_executor_synthesizer_agent",
    description="Retrieves corpus evidence via tools and synthesizes a grounded EssaySynthesisOutput.",
    instruction=EXECUTOR_INSTRUCTION.strip(),
    tools=[search_thesis_literature, list_ingested_papers],
    output_schema=EssaySynthesisOutput,
)

writer_agent = Agent(
    model=_build_gemini_model(
        os.getenv("WRITER_GEMINI_MODEL", DEFAULT_MODEL_NAME),
        WRITER_API_KEY,
    ),
    name="latex_compiler_agent",
    description="Compiles verified EssaySynthesisOutput into structured LaTeX following format and citation style.",
    instruction=WRITER_INSTRUCTION.strip(),
    output_schema=LatexCompilationOutput,
)

# Keep `root_agent` exported for ADK CLI / web tools[cite: 5]
root_agent = executor_agent


# ---------------------------------------------------------------------------
# Python Pipeline Guards (Citation Verification + LaTeX Sanity Check)
# ---------------------------------------------------------------------------

def verify_synthesis_citations(
    synthesis: EssaySynthesisOutput,
    retrieved_chunks_by_id: dict[str, dict[str, Any]],
) -> tuple[EssaySynthesisOutput, dict[str, Any]]:
    """Strips citations whose chunk_id was not returned by retrieval and downgrades ungrounded blocks."""
    total_citations = 0
    valid_citations = 0
    removed_citations: list[dict[str, Any]] = []
    downgraded_blocks = 0

    for block in synthesis.sections:
        original_count = len(block.citations)
        kept = []
        for cit in block.citations:
            total_citations += 1
            chunk_meta = retrieved_chunks_by_id.get(cit.chunk_id)
            if chunk_meta is not None and chunk_meta.get("source_file") == cit.source_file:
                kept.append(cit)
                valid_citations += 1
            else:
                removed_citations.append(
                    {
                        "chunk_id": cit.chunk_id,
                        "source_file": cit.source_file,
                        "reason": "chunk_id_or_source_not_in_retrieved_results",
                    }
                )
        block.citations = kept

        if block.grounding_status == "supported" and not kept:
            block.grounding_status = "insufficient_evidence"
            downgraded_blocks += 1
        elif block.grounding_status == "supported" and original_count > 0 and len(kept) < original_count:
            block.grounding_status = "partial"

    report = {
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "removed_count": len(removed_citations),
        "removed_citations": removed_citations,
        "downgraded_blocks": downgraded_blocks,
    }
    return synthesis, report


def check_latex_output(latex_text: str) -> dict[str, Any]:
    """Performs basic structural and escaping checks on generated LaTeX."""
    brace_depth = 0
    unbalanced_braces = 0
    unescaped_chars: set[str] = set()

    for line in latex_text.splitlines():
        code = line.split("%", 1)[0]
        i = 0
        while i < len(code):
            ch = code[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth < 0:
                    unbalanced_braces += 1
                    brace_depth = 0
            elif ch in {"&", "#"}:
                unescaped_chars.add(ch)
            i += 1

    unbalanced_braces += abs(brace_depth)

    env_stack: list[str] = []
    mismatched_envs: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([A-Za-z0-9_*]+)\}", latex_text):
        kind, env_name = match.group(1), match.group(2)
        if kind == "begin":
            env_stack.append(env_name)
        else:
            if not env_stack or env_stack[-1] != env_name:
                mismatched_envs.append(f"unmatched \\end{{{env_name}}}")
                if env_stack:
                    env_stack.pop()
            else:
                env_stack.pop()
    for leftover in env_stack:
        mismatched_envs.append(f"unclosed \\begin{{{leftover}}}")

    return {
        "is_clean": unbalanced_braces == 0 and not mismatched_envs and not unescaped_chars,
        "unbalanced_braces": unbalanced_braces,
        "mismatched_environments": mismatched_envs,
        "suspicious_unescaped_chars": sorted(unescaped_chars),
    }


# ---------------------------------------------------------------------------
# Sequential 3-Agent Runner with Per-Agent Keys & Tool-Trace Capture
# ---------------------------------------------------------------------------

async def _run_single_agent(
    agent: Agent,
    prompt_text: str,
    user_id: str = "thesis_user",
    session_service: InMemorySessionService | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Runs one ADK agent and captures any retrieved chunks from `search_thesis_literature`."""
    svc = session_service or InMemorySessionService()
    session = await svc.create_session(app_name=APP_NAME, user_id=user_id)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=svc)

    content = types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])
    final_text_parts: list[str] = []
    retrieved_chunks: list[dict[str, Any]] = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        for func_resp in event.get_function_responses():
            resp_data = func_resp.response
            if isinstance(resp_data, dict) and isinstance(resp_data.get("results"), list):
                retrieved_chunks.extend(resp_data["results"])

        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text_parts.append(part.text)

    return "\n".join(final_text_parts).strip(), retrieved_chunks


def _parse_pydantic_json(raw_text: str, model_cls: type[Any]) -> Any:
    """Parses JSON or markdown-fenced JSON into the target Pydantic schema."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return model_cls.model_validate_json(cleaned)


async def run_three_agent_pipeline(
    prompt: str,
    format_profile: str = "thesis_chapter",
    referencing_style: str = "author_year_natbib",
) -> dict[str, Any]:
    """Executes Splitter -> Executor + Citation Guard -> Writer + LaTeX Guard."""
    session_service = InMemorySessionService()

    plan_raw, _ = await _run_single_agent(
        splitter_agent,
        prompt,
        session_service=session_service,
    )
    plan: SubQuestionPlan = _parse_pydantic_json(plan_raw, SubQuestionPlan)

    executor_input = json.dumps(
        {
            "main_prompt": prompt,
            "sub_question_plan": plan.model_dump(),
        },
        ensure_ascii=False,
        indent=2,
    )
    synth_raw, retrieved_chunks = await _run_single_agent(
        executor_agent,
        executor_input,
        session_service=session_service,
    )
    synthesis: EssaySynthesisOutput = _parse_pydantic_json(synth_raw, EssaySynthesisOutput)

    chunks_by_id = {
        str(item["chunk_id"]): item
        for item in retrieved_chunks
        if isinstance(item, dict) and item.get("chunk_id")
    }
    synthesis, citation_audit = verify_synthesis_citations(synthesis, chunks_by_id)

    writer_input = json.dumps(
        {
            "format_profile": format_profile,
            "referencing_style": referencing_style,
            "verified_synthesis": synthesis.model_dump(),
        },
        ensure_ascii=False,
        indent=2,
    )
    latex_raw, _ = await _run_single_agent(
        writer_agent,
        writer_input,
        session_service=session_service,
    )
    latex_out: LatexCompilationOutput = _parse_pydantic_json(latex_raw, LatexCompilationOutput)
    latex_audit = check_latex_output(latex_out.latex_body)

    return {
        "plan": plan,
        "synthesis": synthesis,
        "citation_audit": citation_audit,
        "latex_output": latex_out,
        "latex_audit": latex_audit,
    }


def run_pipeline_sync(
    prompt: str,
    format_profile: str = "thesis_chapter",
    referencing_style: str = "author_year_natbib",
) -> dict[str, Any]:
    """Synchronous entry point for running the 3-agent pipeline."""
    return asyncio.run(
        run_three_agent_pipeline(
            prompt=prompt,
            format_profile=format_profile,
            referencing_style=referencing_style,
        )
    )