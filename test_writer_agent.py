from __future__ import annotations

import asyncio
import copy
import json
import os
import sys

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tools.latex_tools import DEFAULT_TARGET_FILENAME, OUTPUT_DIR
from schemas.schema import LatexCompilationOutput
from agents.writer_agent import writer_agent

INITIAL_SECTION_KEY = "chapter_2_lit_review"
NEW_SECTION_KEY = "chapter_2_peft_data_efficiency"

BASE_SYNTHESIS = {
    "essay_title": "Parameter-Efficient Fine-Tuning and Data Efficiency in Low-Resource ASR",
    "main_prompt": (
        "Synthesize evidence on parameter-efficient fine-tuning (PEFT) versus full fine-tuning "
        "and data scaling in low-resource automatic speech recognition (ASR), including relevance "
        "to Kom ASR."
    ),
    "answered_sub_questions": [
        "How do PEFT methods compare to full fine-tuning in low-resource ASR?",
        "What evidence exists on speech data scaling and Kom ASR?",
    ],
    "sections": [
        {
            "heading": "PEFT vs. Full Fine-Tuning in Low-Resource Speech",
            "grounding_status": "supported",
            "argument_text": (
                "Parameter-efficient fine-tuning reduces trainable parameters during adaptation of "
                "pretrained speech models while preserving most backbone weights. Controlled "
                "comparisons on low-resource African languages and multilingual speech models "
                "examine trade-offs between Word Error Rate (WER), trainable parameter counts, "
                "and adaptation efficiency."
            ),
            "citations": [
                {
                    "chunk_id": "imam2026_p1_c0",
                    "doc_id": "imam-etal-2026-full",
                    "source_file": "imam-etal-2026-full.pdf",
                    "page": "1",
                    "section": "Introduction",
                    "supporting_excerpt": (
                        "Compares full fine-tuning with parameter-efficient adaptation on "
                        "Whisper-Small for low-resource African ASR."
                    ),
                },
                {
                    "chunk_id": "hu2021_p1_c0",
                    "doc_id": "hu2021lora",
                    "source_file": "hu2021lora.pdf",
                    "page": "1",
                    "section": "Abstract",
                    "supporting_excerpt": (
                        "Introduces Low-Rank Adaptation (LoRA) by freezing pretrained weights "
                        "and injecting trainable rank decomposition matrices."
                    ),
                },
            ],
            "caveat_or_gap": None,
        },
        {
            "heading": "Data Efficiency and Kom Speech Resources",
            "grounding_status": "partial",
            "argument_text": (
                "Field-collected speech resources for Kom have demonstrated the feasibility of "
                "applying pretrained multilingual speech models in low-resource settings, while "
                "cross-lingual data-scaling studies show that labelled speech volume strongly "
                "affects recognition performance across languages."
            ),
            "citations": [
                {
                    "chunk_id": "faytak2026_p1_c0",
                    "doc_id": "faytak-etal-2026-short",
                    "source_file": "faytak-etal-2026-short.pdf",
                    "page": "1",
                    "section": "Introduction",
                    "supporting_excerpt": (
                        "Investigates short-form verbal arts speech data for Kom and ASR feasibility."
                    ),
                },
                {
                    "chunk_id": "akera2026_p1_c0",
                    "doc_id": "pmlr-v314-akera26b",
                    "source_file": "pmlr-v314-akera26b.pdf",
                    "page": "1",
                    "section": "Introduction",
                    "supporting_excerpt": (
                        "Evaluates speech data scaling for ASR in low-resource African languages."
                    ),
                },
            ],
            "caveat_or_gap": (
                "Controlled PEFT-by-data-regime experiments specifically on Kom remain an open area "
                "to evaluate directly."
            ),
        },
    ],
    "cross_paper_tensions_and_gaps": [
        "Relative trade-offs between PEFT and full fine-tuning vary across model families, "
        "languages, and labelled data budgets."
    ],
    "unanswered_or_low_evidence_topics": [
        "Edge-device latency and energy consumption for adapted Kom ASR."
    ],
}

SECOND_SECTION_SYNTHESIS = copy.deepcopy(BASE_SYNTHESIS)
SECOND_SECTION_SYNTHESIS["sections"] = [
    {
        "heading": "Depth-Aware Adaptation and Layer Specialization",
        "grounding_status": "supported",
        "argument_text": (
            "Recent multilingual speech adaptation studies examine whether parameter updates "
            "behave differently across transformer layers and whether layer-selective or "
            "depth-aware adaptation improves efficiency under low-resource constraints."
        ),
        "citations": [
            {
                "chunk_id": "xiao2026_p1_c0",
                "doc_id": "xiao-etal-2026-adapting",
                "source_file": "xiao-etal-2026-adapting.pdf",
                "page": "1",
                "section": "Introduction",
                "supporting_excerpt": (
                    "Investigates depth-aware adaptation for multilingual speech recognition in "
                    "low-resource languages."
                ),
            },
            {
                "chunk_id": "kim2024_p1_c0",
                "doc_id": "kim24s_interspeech",
                "source_file": "kim24s_interspeech.pdf",
                "page": "1",
                "section": "Introduction",
                "supporting_excerpt": (
                    "Examines convolution-augmented parameter-efficient adaptation for speech "
                    "recognition."
                ),
            },
        ],
        "caveat_or_gap": None,
    }
]


async def run_single_case(
    runner: Runner,
    user_id: str,
    session_id: str,
    label: str,
    section_key: str,
    synthesis_payload: dict,
) -> LatexCompilationOutput:
    print(f"\n==================== {label} (section_key={section_key}) ====================")
    payload = {
        "format_profile": "thesis_chapter",
        "referencing_style": "biblatex_authoryear",
        "target_filename": DEFAULT_TARGET_FILENAME,
        "section_key": section_key,
        "include_preamble_and_cover": True,
        "verified_synthesis": synthesis_payload,
    }

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=json.dumps(payload, indent=2))],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    print(f"[TOOL CALL] {fc.name} -> args keys: {list((fc.args or {}).keys())}")
                if getattr(part, "function_response", None):
                    fr = part.function_response
                    print(f"[TOOL RESP] {fr.name} -> {fr.response}")

        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(
                p.text for p in event.content.parts if getattr(p, "text", None)
            ).strip()

    parsed = LatexCompilationOutput.model_validate_json(final_text)
    print(f"[FORMATTING NOTES] {parsed.formatting_notes}")
    print(f"[REFERENCED CHUNK IDS] {parsed.referenced_chunk_ids}")
    return parsed


async def main() -> None:
    app_name = "writer_incremental_test"
    user_id = "test_user"
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=app_name, user_id=user_id)
    runner = Runner(agent=writer_agent, app_name=app_name, session_service=session_service)

    target_path = OUTPUT_DIR / DEFAULT_TARGET_FILENAME
    print(f"Target LaTeX file: {target_path}")

    # Test 1: Append a new section to the existing file (or initialize if not yet created)
    await run_single_case(
        runner=runner,
        user_id=user_id,
        session_id=session.id,
        label="TEST 1: APPEND NEW SECTION",
        section_key=NEW_SECTION_KEY,
        synthesis_payload=SECOND_SECTION_SYNTHESIS,
    )

    # Test 2: Re-send an already-present section_key to verify duplicate rejection / skip
    await run_single_case(
        runner=runner,
        user_id=user_id,
        session_id=session.id,
        label="TEST 2: DUPLICATE SECTION CHECK",
        section_key=INITIAL_SECTION_KEY,
        synthesis_payload=BASE_SYNTHESIS,
    )

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())