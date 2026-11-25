from pathlib import Path
import chromadb
import pytest

from . import ingest


class _FakePage:
  def __init__(self, text: str):
    self._text = text

  def extract_text(self) -> str:
    return self._text


class _FakePdfReader:
  def __init__(self, pages_text: list[str]):
    self.pages = [_FakePage(t) for t in pages_text]


def _install_fake_pdf_reader(monkeypatch: pytest.MonkeyPatch, pages_by_name: dict[str, list[str]]) -> None:
  def _factory(path_str: str):
    name = Path(path_str).name
    if name not in pages_by_name:
      raise AssertionError(f"Unexpected PDF path in test: {path_str}")
    return _FakePdfReader(pages_by_name[name])

  monkeypatch.setattr(ingest, "PdfReader", _factory)


def test_clean_page_text_dehyphenates_and_preserves_casing():
  raw = (
      "Multi-\nlingual ASR   models   for Kom\n\n"
      "Second   paragraph   with   WordErrorRate   and   LoRA."
  )
  cleaned = ingest._clean_page_text(raw)

  assert "Multilingual ASR models for Kom" in cleaned
  assert "WordErrorRate" in cleaned
  assert "LoRA" in cleaned
  assert "\n\n" in cleaned


@pytest.mark.parametrize(
    "line,expected",
    [
        ("1. Introduction", "1. Introduction"),
        ("2.1 Methodology", "2.1 Methodology"),
        ("EXPERIMENTAL SETUP", "EXPERIMENTAL SETUP"),
        ("This is a regular sentence that ends with a period.", None),
        ("This line is much too long to be a clean academic heading " + " ".join(["word"] * 15), None),
    ],
)
def test_detect_heading(line: str, expected: str | None):
  assert ingest._detect_heading(line) == expected


def test_word_chunks_respect_budget_overlap_and_merge_tiny_tail():
  # 25 words with target=10 and overlap=2 -> first chunk has 10 words,
  # second starts 2 words back and takes remaining words without leaving a tiny 1-word scrap.
  text = " ".join([f"w{i}" for i in range(25)])
  chunks = ingest._split_text_into_word_chunks(text, chunk_words=10, overlap_words=2)

  assert len(chunks) >= 2
  first_words = chunks[0].split()
  second_words = chunks[1].split()

  assert len(first_words) == 10
  # Overlap of 2 words from end of first chunk appears at start of second chunk
  assert second_words[:2] == first_words[-2:]
  for ch in chunks:
    assert ingest.count_words(ch) >= 5


def test_extract_and_chunk_pdf_metadata_and_section_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  pdf_path = tmp_path / "kom_asr_paper.pdf"
  pdf_path.write_bytes(b"%PDF-fake")

  page1 = "1. Introduction\n\n" + " ".join(["token"] * 45)
  page2 = " ".join(["followup"] * 30)
  _install_fake_pdf_reader(monkeypatch, {"kom_asr_paper.pdf": [page1, page2]})

  records, total_pages = ingest.extract_and_chunk_pdf(pdf_path, chunk_words=25, overlap_words=5)

  assert total_pages == 2
  assert len(records) >= 2

  for idx, rec in enumerate(records):
    meta = rec["metadata"]
    assert meta["doc_id"] == "kom_asr_paper"
    assert meta["source_file"] == "kom_asr_paper.pdf"
    assert meta["page"] in (1, 2)
    assert meta["chunk_index"] >= 0
    assert meta["section"] == "1. Introduction"
    assert meta["word_count"] == ingest.count_words(rec["document"].split("\n", 1)[1])
    assert meta["doc_type"] == ingest.DEFAULT_DOC_TYPE
    assert meta["access_level"] == ingest.DEFAULT_ACCESS_LEVEL
    assert meta["ingested_at"]
    assert rec["document"].startswith(
        f"[Document: kom_asr_paper.pdf | Page: {meta['page']} | Section: 1. Introduction]\n"
    )
    assert rec["id"].startswith("kom_asr_paper_p")


def test_ingest_papers_idempotent_and_handles_empty_or_scanned_pdfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
  corpus_dir = tmp_path / "papers_corpus"
  corpus_dir.mkdir()
  db_dir = tmp_path / "papers_db"

  # 1. Empty corpus directory prints helpful message and creates nothing extra
  monkeypatch.setattr(ingest, "DB_PATH", str(db_dir))
  monkeypatch.setattr(ingest, "COLLECTION_NAME", "test_literature_corpus")

  ingest.ingest_papers(corpus_dir=corpus_dir)
  out_empty = capsys.readouterr().out
  assert "No PDF files found" in out_empty

  # 2. Add one valid PDF and one scanned/empty PDF
  (corpus_dir / "valid_paper.pdf").write_bytes(b"%PDF-fake")
  (corpus_dir / "scanned_paper.pdf").write_bytes(b"%PDF-fake")
  _install_fake_pdf_reader(
      monkeypatch,
      {
          "valid_paper.pdf": ["2.1 Methodology\n\n" + " ".join(["ASR"] * 30)],
          "scanned_paper.pdf": ["   ", ""],
      },
  )

  ingest.ingest_papers(corpus_dir=corpus_dir, chunk_words=20, overlap_words=4)
  out_first = capsys.readouterr().out
  assert "No extractable text found in scanned_paper.pdf" in out_first

  client = chromadb.PersistentClient(path=str(db_dir))
  collection = client.get_or_create_collection(name="test_literature_corpus")
  first_count = collection.count()
  assert first_count > 0

  # 3. Re-running ingestion must be idempotent (no duplicate chunks)
  ingest.ingest_papers(corpus_dir=corpus_dir, chunk_words=20, overlap_words=4)
  assert collection.count() == first_count