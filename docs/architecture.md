# System Architecture: Autonomous Academic Literature Synthesis Agent

## 1. High-Level Architectural Overview

The **Autonomous Academic Literature Synthesis Agent** follows a decoupled **Retrieval-Augmented Generation (RAG) Agent Architecture** built on the **Google Agent Development Kit (`google-adk`)** and **ChromaDB**. Rather than relying on parametric pre-training weights, the system separates knowledge persistence from reasoning, forcing the Large Language Model (LLM) to act strictly as an analytical synthesizer over retrieved primary literature.

```text
+-----------------------------------------------------------------------------------+
|                        PHASE 1: OFFLINE INGESTION PIPELINE                        |
|                                                                                   |
|  [papers_corpus/*.pdf] ---> [PDF Text Extractor] ---> [Recursive Text Chunker]    |
|                                                                  |                |
|                                                                  v                |
|  [papers_db/ (ChromaDB)] <--- [Metadata Tagger] <--- [Dense Vector Embeddings]    |
+-----------------------------------------------------------------------------------+
                                         |
                                         | (Persistent Local Vector Index)
                                         v
+-----------------------------------------------------------------------------------+
|                        PHASE 2: ONLINE AGENTIC RAG RUNTIME                        |
|                                                                                   |
|  [Researcher Query] ---> [Google ADK Agent (Gemini)]                              |
|                                |             ^                                    |
|     (1. Autonomous Tool Call)  |             |  (3. Grounded Context + Citations) |
|                                v             |                                    |
|                     [search_thesis_literature Tool]                               |
|                                |             ^                                    |
|      (2. Semantic Query)       |             |  (Top-K Ranked Chunks)             |
|                                v             |                                    |
|                          [ChromaDB Collection]                                    |
|                                                                                   |
|  [Google ADK Agent] ---> [Strict Grounding & Attribution Guardrails]              |
|                                |                                                  |
|                                v                                                  |
|              [Synthesized Literature Review w/ Citations]                         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core System Components

### 2.1 Data Ingestion & Indexing Layer (`literature_agent/ingest.py`)
Responsible for transforming unstructured academic PDF files into searchable semantic representations:
* **Document Parser (`pypdf`):** Iterates through the local `papers_corpus/` directory, extracting raw text page-by-page while preserving page indices.
* **Context-Preserving Chunker:** Splits extracted text into overlapping segments (e.g., $\text{chunk\_size} = 1000$ characters, $\text{overlap} = 200$ characters) so that cross-sentence methodological descriptions and equations are not severed at chunk boundaries.
* **Metadata Enrichment:** Attaches structured metadata (`source_file`, `page_number`, `chunk_id`) to every text chunk to guarantee verifiable traceability during synthesis.
* **Vector Store (`ChromaDB`):** Computes dense vector embeddings and persists them locally in `papers_db/` under the `thesis_literature_corpus` collection.

### 2.2 Semantic Retrieval Tool Layer (`literature_agent/tools.py`)
Exposes deterministic Python functions to the ADK runtime via clean type annotations and docstrings:
* **`search_thesis_literature(query_topic: str)`:** Converts the agent's formulated search query into a vector embedding, executes cosine similarity search against `ChromaDB`, retrieves the top-$k$ most relevant chunks, and formats each excerpt with explicit `--- Source: <filename> (Page X) ---` headers.

### 2.3 Agent Orchestration & Guardrail Layer (`literature_agent/agent.py`)
Powered by `google-adk` and `gemini-3.5-flash`, this layer governs reasoning, multi-step retrieval, and citation enforcement:
* **Autonomous Query Formulation:** Decomposes complex user prompts (e.g., comparing two architectures across multiple papers) into targeted semantic search queries.
* **Multi-Turn Retrieval Loop:** Capable of invoking `search_thesis_literature` multiple times if initial retrieved chunks lack specific baseline metrics or experimental parameters.
* **Negative Constraint Guardrails:** Strict system instructions prohibit the model from injecting external pre-training knowledge or fabricating citations. If evidence is absent from the retrieved context, the agent must explicitly report the gap.

---

## 3. End-to-End Execution Flow

1. **Corpus Preparation (Offline):** The researcher places target `.pdf` publications into `papers_corpus/` and executes `python -m literature_agent.ingest`.
2. **Query Submission (Online):** The researcher submits an analytical question via `adk run` or `adk web` (e.g., *"Compare the Word Error Rates and fine-tuning strategies across the low-resource ASR papers"*).
3. **Tool Invocation:** The ADK runtime recognizes the need for primary literature evidence and invokes `search_thesis_literature`.
4. **Vector Similarity Search:** ChromaDB computes semantic similarity between the query vector and stored document embeddings, returning the top-$k$ ranked excerpts along with their source filenames and page numbers.
5. **Grounded Synthesis:** The LLM synthesizes the retrieved excerpts into a structured academic comparison, appending inline source citations to every claim, metric, and methodological statement.

---

## 4. Design Trade-offs & Scalability Considerations

* **Local Persistence vs. Cloud Vector Stores:** Using local persistent `ChromaDB` eliminates external cloud latency, avoids API costs during rapid thesis experimentation, and keeps unpublished research drafts private.
* **Chunk Size vs. Context Precision:** Smaller chunks improve retrieval precision for specific numbers (e.g., hyperparameters or metrics), whereas larger overlapping chunks preserve broader experimental context. Overlapping windows balance both requirements.
* **Deterministic Tool Boundaries:** Isolating database logic inside `tools.py` allows the retrieval layer to be independently unit-tested with `pytest` and upgraded (e.g., adding hybrid BM25 keyword + dense vector search) without modifying agent instructions.