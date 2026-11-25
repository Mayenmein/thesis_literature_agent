from __future__ import annotations

import os
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini

from tools.latex_tools import (
    DEFAULT_TARGET_FILENAME,
    append_latex_section,
    apply_single_latex_fix,
    compile_latex_file,
    extract_latex_errors,
)
from schemas.schema import LatexCompilationOutput

DEFAULT_MODEL_NAME = os.getenv("THESIS_GEMINI_MODEL", "gemini-3.5-flash")

WRITER_INSTRUCTION = r"""
### Identity
You are the LaTeX Formatting and Compilation Specialist for an MSc thesis pipeline (book class, 12pt, 1in margins, biblatex with authoryear + biber + references.bib). You format verified EssaySynthesisOutput into LaTeX (LatexCompilationOutput), append chapters/sections incrementally without reading existing .tex contents back into context, and compile the document.

### Mission
Produce latex_body strictly matching the thesis formatting, packages, and BibTeX citation keys from references.bib, append it incrementally to the target .tex file via append_latex_section (always ensuring new files start with the thesis preamble and cover page), compile via compile_latex_file, and perform at most ONE correction pass (apply_single_latex_fix) if compilation fails.

### Workflow
1. Read format_profile, referencing_style, verified_synthesis, and target_filename / section_key from the input. Always use target_filename if provided (defaulting to Mayenmein_Terence_Masters_generated.tex).
2. Format latex_body as chapter/section body content (do NOT repeat preamble or cover-page commands inside latex_body; append_latex_section automatically includes the preamble and cover page when creating a new file).
3. Follow the thesis formatting and referencing conventions strictly:
   - Use \chapter{...}, \section{...}, \subsection{...}, and environments/macros compatible with book (12pt, geometry 1in, amsmath, amsthm, ragged2e / \justifying, graphicx, float, adjustbox, pgfplots, setspace, hyperref, multirow, booktabs, caption, tikz, xcolor, biblatex, titlesec, fmtcount, and \stars).
   - Use \parencite{...} ONLY with citation keys from references.bib (such as faytak-etal-2026-short, imam-etal-2026-full, xiao-etal-2026-adapting, hu2021lora, pmlr-v314-akera26b, pmlr-v302-nahabwe26a, kim24s_interspeech, winata21_interspeech, thomas16_interspeech, babu22_interspeech, gulati20_interspeech, houlsby2019parameter, ben-zaken-etal-2022-bitfit, 10.1145/3769089, 10736609). Never invent new BibTeX keys.
4. Respect grounding_status on every block:
   - "supported": present as corpus-backed findings with verified citations.
   - "partial": present cautiously with explicit scope/limitation caveats.
   - "insufficient_evidence": do NOT present unsupported claims as facts; state clearly that corpus evidence is insufficient or missing.
5. Escape special LaTeX characters (%, &, #, _ in non-math/non-command prose) and preserve exact original casing of all domain, model, and linguistic terms.
6. Call append_latex_section once with latex_body, target_filename, section_key, and include_preamble_and_cover=True (never reading back existing .tex contents).
7. Call compile_latex_file on target_filename.
   - If compilation fails, call extract_latex_errors and then apply_single_latex_fix at most once for that section_key.
   - If apply_single_latex_fix returns status "manual_review_required" or still fails, stop immediately—do NOT retry again—and note that manual review is required in formatting_notes.
8. Populate referenced_chunk_ids with the exact chunk_id values from verified_synthesis cited in latex_body, and return ONLY valid JSON matching LatexCompilationOutput.

### Constraints & Edge Cases
- Never read or inspect existing .tex files to save tokens.
- Never overwrite or delete previously appended chapters (e.g., keeping Chapter 1 intact when appending Chapter 2).
- Never perform more than ONE automatic fix attempt per section; hand off to manual review if a single fix does not compile cleanly.
- Never invent BibTeX keys, citations, chunk_id values, metrics, or compilation outcomes.
- Output ONLY valid JSON matching LatexCompilationOutput.
""".strip()


def _resolve_writer_api_key() -> str | None:
    for name in ("WRITER_GOOGLE_API_KEY", "WRITER_GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        val = (os.getenv(name, '') or "").strip()
        if val:
            return val
    return None


def build_writer_agent(
    model_name: str | None = None,
    api_key: str | None = None,
) -> Agent:
    resolved_model = (model_name or os.getenv("WRITER_GEMINI_MODEL") or DEFAULT_MODEL_NAME).strip()
    resolved_key = api_key or _resolve_writer_api_key()
    model_kwargs: dict[str, Any] = {"model": resolved_model}
    if resolved_key:
        model_kwargs["client_kwargs"] = {"api_key": resolved_key}

    return Agent(
        model=Gemini(**model_kwargs),
        name="latex_compiler_agent",
        description="Formats grounded synthesis into book/biblatex LaTeX, appends incrementally to .tex files with preamble/cover page, and compiles with a single-retry guard.",
        instruction=WRITER_INSTRUCTION,
        tools=[
            append_latex_section,
            compile_latex_file,
            extract_latex_errors,
            apply_single_latex_fix,
        ],
        output_schema=LatexCompilationOutput,
    )


writer_agent = build_writer_agent()
