# Problem Statement: Autonomous Academic Literature Synthesis Agent

## 1. Background & Context
Conducting a rigorous, comprehensive literature review is a foundational requirement of academic research and Master of Science (MSc) thesis execution. Researchers must identify relevant publications, extract experimental setups, compare baseline metrics, and synthesize methodological trade-offs across dozens—or hundreds—of technical papers. As the volume of published scientific literature grows exponentially, manual review workflows have become increasingly unsustainable.

## 2. Core Challenges

### 2.1 Manual Screening and Synthesis Bottlenecks
Manually reading, annotating, and cross-referencing dense technical PDFs introduces severe cognitive overload and delays early-stage research execution. Tracking specific hyperparameters, dataset splits, and evaluation metrics across multiple studies in spreadsheets or static notes is error-prone and difficult to scale.

### 2.2 Limitations of Lexical Keyword Search
Traditional academic search tools and local PDF readers rely primarily on exact keyword matching. When different authors use varied terminology to describe conceptually identical techniques (e.g., *parameter-efficient fine-tuning* vs. *low-rank adaptation*), lexical search fails to surface critical connections, leading to incomplete literature coverage and missed baseline comparisons.

### 2.3 Hallucination and Attribution Risks in General-Purpose LLMs
While general-purpose Large Language Models (LLMs) can rapidly summarize text, relying on them directly for academic synthesis introduces unacceptable risks:
* **Parametric Hallucinations:** LLMs frequently fabricate paper titles, authors, experimental results, or statistical claims when queried beyond their immediate context.
* **Lack of Verifiable Grounding:** Standard chat interfaces blend parametric pre-training knowledge with user-provided text, making it difficult to trace whether a claim originates from the researcher's curated corpus or external model assumptions.

## 3. Proposed Solution
To overcome these limitations, this project proposes the design and implementation of an **Autonomous Academic Literature Synthesis Agent** built using the **Google Agent Development Kit (`google-adk`)** and **Retrieval-Augmented Generation (RAG)**. 

By indexing a local repository of research PDFs into a semantic vector database (**ChromaDB**) and equipping an LLM agent with constrained retrieval tools and strict prompt guardrails, the system will enable researchers to query, compare, and synthesize multi-paper evidence with verifiable source citations.

## 4. Primary Objectives
1. **Automated Corpus Ingestion & Semantic Indexing:** Build a deterministic pipeline to parse academic PDFs, segment text into context-preserving semantic chunks, and store embeddings with file-level metadata in a persistent local vector store.
2. **Agentic Cross-Document Retrieval:** Enable the agent to autonomously formulate semantic search queries, retrieve relevant methodological excerpts across multiple papers, and synthesize comparative analyses.
3. **Strict Citation Grounding & Zero-Hallucination Guardrails:** Enforce negative constraints within the agent's instruction architecture so that all generated claims, metrics, and comparisons are strictly grounded in retrieved document chunks with explicit source file attribution.

## 5. Success Criteria
* **Retrieval Precision:** The vector search pipeline accurately surfaces relevant paper excerpts within the top-$k$ retrieved chunks for domain-specific research queries.
* **Faithfulness & Grounding:** Generated syntheses contain zero fabricated citations or unverified external claims, explicitly stating when the local corpus lacks sufficient evidence to answer a query.
* **Workflow Acceleration:** Significant reduction in the time required to extract comparative baseline metrics and methodological summaries across multi-paper repositories.