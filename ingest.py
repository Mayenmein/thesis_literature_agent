from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

# --- Configuration ---
CORPUS_DIR = Path("./papers_corpus")
DB_PATH = "./papers_db"
COLLECTION_NAME = "thesis_literature_corpus"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Word-based chunking parameters (tune to match your corpus & embedding window)
CHUNK_WORDS = 400         # Target words per chunk (e.g., 300-600 words for prose)
CHUNK_OVERLAP_WORDS = 50  # ~12.5% word overlap between consecutive chunks
DEFAULT_DOC_TYPE = "research_paper"
DEFAULT_ACCESS_LEVEL = "internal_research"

WORD_PATTERN = re.compile(r"\S+")


def count_words(text: str) -> int:
    """Counts whitespace-delimited word tokens in a text string."""
    if not text:
        return 0
    return len(WORD_PATTERN.findall(text))


def _clean_page_text(raw_text: str) -> str:
    """Normalizes line breaks and horizontal whitespace while keeping paragraph breaks."""
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Join hyphenated line breaks across wrapped lines (e.g., "multi-\nlingual" -> "multilingual")
    normalized = re.sub(r"([A-Za-zÀ-ÖØ-öø-ÿ])-\s*\n\s*([A-Za-zÀ-ÖØ-öø-ÿ])", r"\1\2", normalized)

    cleaned_lines: list[str] = []
    for line in normalized.split("\n"):
        stripped_line = re.sub(r"[ \t]+", " ", line).strip()
        if stripped_line:
            cleaned_lines.append(stripped_line)
        else:
            # Preserve blank lines as paragraph boundaries
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")

    # Collapse 2+ consecutive blank lines into a single paragraph break
    text_with_paras = "\n".join(cleaned_lines)
    text_with_paras = re.sub(r"\n{3,}", "\n\n", text_with_paras)
    return text_with_paras.strip()


def _detect_heading(paragraph: str) -> str | None:
    """Best-effort heuristic to detect short section headings in academic PDFs."""
    first_line = paragraph.split("\n", 1)[0].strip()
    if not first_line:
        return None
    # Skip long sentences or lines ending with terminal punctuation
    if len(first_line) > 100 or first_line.endswith((".", "?", "!", ";", ",")):
        return None

    words = first_line.split()
    if len(words) > 14:
        return None

    # Common numbered sections (e.g., "1. Introduction", "2.1 Methodology", "III. RESULTS")
    if re.match(r"^(\d+(\.\d+)*|[IVXLC]+)\s*[\.:)]?\s+\S+", first_line):
        return first_line

    # Common academic section titles
    common_sections = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "literature review",
        "methodology",
        "methods",
        "experimental setup",
        "experiments",
        "results",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
        "references",
        "acknowledgments",
        "appendix",
    }
    if first_line.lower() in common_sections:
        return first_line

    # Title-cased or ALL-CAPS short line without terminal period
    if first_line.isupper() and len(words) >= 2:
        return first_line.title()

    return None


def _split_text_into_word_chunks(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Splits text into overlapping chunks measured by word count."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be greater than 0.")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be >= 0 and strictly less than chunk_words.")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_spans: list[tuple[str, int, int]] = []
    current_total_words = 0

    for para in paragraphs:
        para_words = para.split()
        p_len = len(para_words)
        if p_len == 0:
            continue

        start = 0
        while start < p_len:
            remaining_budget = chunk_words - current_total_words
            if remaining_budget <= 0:
                chunk_text = _assemble_chunk_from_spans(current_spans)
                if chunk_text:
                    chunks.append(chunk_text)
                current_spans, current_total_words = _build_overlap_spans(
                    current_spans, overlap_words
                )
                remaining_budget = chunk_words - current_total_words

            take = min(p_len - start, remaining_budget)
            end = start + take
            slice_text = " ".join(para_words[start:end])

            current_spans.append((slice_text, 0, take))
            current_total_words += take
            start = end

            if current_total_words >= chunk_words:
                chunk_text = _assemble_chunk_from_spans(current_spans)
                if chunk_text:
                    chunks.append(chunk_text)
                current_spans, current_total_words = _build_overlap_spans(
                    current_spans, overlap_words
                )

    if current_spans:
        chunk_text = _assemble_chunk_from_spans(current_spans)
        if chunk_text:
            if (
                chunks
                and count_words(chunk_text) < max(25, int(0.2 * chunk_words))
                and count_words(chunks[-1]) + count_words(chunk_text) <= int(1.25 * chunk_words)
            ):
                chunks[-1] = f"{chunks[-1]}\n\n{chunk_text}"
            else:
                chunks.append(chunk_text)

    return chunks


def _assemble_chunk_from_spans(spans: list[tuple[str, int, int]]) -> str:
    """Joins text pieces into a clean chunk string."""
    parts = [s[0].strip() for s in spans if s[0] and s[0].strip()]
    return "\n\n".join(parts).strip()


def _build_overlap_spans(
    spans: list[tuple[str, int, int]], overlap_words: int
) -> tuple[list[tuple[str, int, int]], int]:
    """Extracts the trailing `overlap_words` from existing spans to seed the next chunk."""
    if overlap_words <= 0 or not spans:
        return [], 0

    all_words: list[str] = []
    for text_piece, _, _ in spans:
        all_words.extend(text_piece.split())

    if not all_words:
        return [], 0

    overlap_slice = all_words[-overlap_words:]
    overlap_text = " ".join(overlap_slice)
    return [(overlap_text, 0, len(overlap_slice))], len(overlap_slice)


def extract_and_chunk_pdf(
    pdf_path: Path,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> tuple[list[dict[str, Any]], int]:
    """Extracts text page-by-page from a PDF, tracks section headings, splits into

    word-based chunks, and prefixes each chunk with provenance/section context.
    Returns (chunk_records, total_pages).
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    chunk_records: list[dict[str, Any]] = []
    ingested_at = datetime.now(timezone.utc).isoformat()
    current_section = "General"

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned_text = _clean_page_text(raw_text)
        if not cleaned_text:
            continue

        paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
        for para in paragraphs:
            detected = _detect_heading(para)
            if detected:
                current_section = detected

        page_chunks = _split_text_into_word_chunks(
            cleaned_text,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )

        for chunk_idx, chunk_body in enumerate(page_chunks):
            chunk_id = f"{pdf_path.stem}_p{page_idx + 1}_c{chunk_idx}"
            word_count = count_words(chunk_body)

            context_prefix = (
                f"[Document: {pdf_path.name} | Page: {page_idx + 1} | "
                f"Section: {current_section}]"
            )
            enriched_document = f"{context_prefix}\n{chunk_body}"

            chunk_records.append({
                "id": chunk_id,
                "document": enriched_document,
                "metadata": {
                    "doc_id": pdf_path.stem,
                    "source_file": pdf_path.name,
                    "page": page_idx + 1,
                    "chunk_index": chunk_idx,
                    "section": current_section,
                    "word_count": word_count,
                    "doc_type": DEFAULT_DOC_TYPE,
                    "access_level": DEFAULT_ACCESS_LEVEL,
                    "ingested_at": ingested_at,
                },
            })

    return chunk_records, total_pages


def ingest_papers(
    corpus_dir: Path = CORPUS_DIR,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
) -> None:
    """Scans the corpus directory for PDFs, chunks them by word count, and upserts into ChromaDB."""
    if not corpus_dir.exists():
        corpus_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created '{corpus_dir}'. Place your research PDFs inside and re-run.")
        return

    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{corpus_dir}'.")
        return

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    total_chunks = 0
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}...")
        try:
            records, total_pages = extract_and_chunk_pdf(
                pdf_path,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )
            if not records:
                print(
                    f"  [Warning] No extractable text found in {pdf_path.name} "
                    "(may be scanned/image-only)."
                )
                continue

            collection.upsert(
                ids=[r["id"] for r in records],
                documents=[r["document"] for r in records],
                metadatas=[r["metadata"] for r in records],
            )
            total_chunks += len(records)
            print(
                f"  -> Indexed {len(records)} word-chunks ({chunk_words}w target, "
                f"{overlap_words}w overlap) from {total_pages} pages."
            )

        except Exception as e:
            print(f"  [Error] Failed to process {pdf_path.name}: {e}")

    print(
        f"\nIngestion complete. Total chunks upserted into '{COLLECTION_NAME}' "
        f"using '{embedding_model_name}': {total_chunks}"
    )


if __name__ == "__main__":
    ingest_papers()