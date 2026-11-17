# Evaluation Plan: Autonomous Academic Literature Synthesis Agent

## 1. Evaluation Philosophy

To validate the reliability and academic rigor of the **Autonomous Academic Literature Synthesis Agent**, evaluation is decoupled into two distinct layers—mirroring the **RAG Triad** framework:

1. **Component-Level Retrieval Evaluation:** Measuring whether the vector index (`ChromaDB`) surfaces the exact source passages required to answer a technical query.

2. **End-to-End Generation & Grounding Evaluation:** Measuring whether the `google-adk` agent synthesizes answers faithfully from retrieved context without parametric hallucinations or fabricated citations.

## 2. Benchmark Dataset Construction ("Golden Set")

Before running quantitative benchmarks, construct a curated **Golden Evaluation Dataset** (`tests/eval_dataset.json`) derived directly from your local `papers_corpus/` repository.

### Dataset Schema

Each entry ($N = 15\text{ to }25$ test cases) contains:

* **`query`:** A realistic thesis research question (single-paper lookup, cross-paper comparison, or out-of-scope negative test).

* **`expected_sources`:** List of target `.pdf` filenames containing the ground-truth answer.

* **`ground_truth_facts`:** Key metrics, model names, or methodological claims expected in the final synthesis.

* **`query_type`:** Categorized into `factual_extraction`, `cross_paper_synthesis`, or `unanswerable_negative`.

## 3. Evaluation Metrics & Mathematical Definitions

### 3.1 Retrieval Layer Metrics (`ChromaDB` Vector Search)

1. **Hit Rate @** $k$ **(**$\text{HR}@k$**):**
   Measures the proportion of queries for which at least one ground-truth source paper appears within the top-$k$ retrieved chunks ($k = 3$ or $k = 5$):
   

   $$
   \text{HR}@k = \frac{1}{\vert{}Q\vert{}} \sum_{q \in Q} \mathbb{I}(\text{Rank}_q \le k)
   $$

   
   *Target Threshold:* $\text{HR}@5 \ge 0.85$ ($85\%$).

2. **Mean Reciprocal Rank (**$\text{MRR}$**):**
   Evaluates how high the first relevant paper ranks in the retrieved results:
   

   $$
   \text{MRR} = \frac{1}{\vert{}Q\vert{}} \sum_{q \in Q} \frac{1}{\text{Rank}_q}
   $$

   
   *Target Threshold:* $\text{MRR} \ge 0.70$.

### 3.2 Generation & Grounding Metrics (Agent Synthesis)

1. **Faithfulness Score (Zero-Hallucination Check):**
   Measures the ratio of atomic claims in the agent's generated synthesis that can be directly verified against the retrieved context chunks:
   

   $$
   \text{Faithfulness} = \frac{\text{Number of Claims Supported by Retrieved Context}}{\text{Total Number of Claims in Response}}
   $$

   
   *Target Threshold:* $\text{Faithfulness} \ge 0.95$ ($95\%$).

2. **Citation Attribution Accuracy:**
   Verifies that every quantitative metric or methodological claim includes a valid `source_file` reference matching the metadata of the retrieved chunks, with zero fabricated filenames.
   *Target Threshold:* $100\%$ valid source mapping.

3. **Abstention Accuracy (Negative Guardrail Test):**
   Evaluates how reliably the agent refuses to answer queries about topics absent from `papers_corpus/` (e.g., *"What does the corpus say about quantum error correction?"*).
   *Target Threshold:* $100\%$ explicit refusal without falling back to general pre-training knowledge.

## 4. Testing & Execution Strategy

### 4.1 Deterministic Unit & Integration Tests (`pytest`)

Located in `tests/test_tools.py` and `tests/test_agent_integration.py`:

* **Mocked DB Tests:** Verify formatting, error handling when `papers_db/` is empty or locked, and metadata parsing without requiring live LLM calls.

* **Schema & Routing Tests:** Confirm that the ADK agent autonomously invokes `search_thesis_literature` when presented with domain queries and skips tool invocation for off-topic greetings.

### 4.2 Automated LLM-as-a-Judge Evaluation (`tests/run_rag_eval.py`)

An automated evaluation script executes all Golden Set queries against the agent, captures both the `retrieved_context` and `final_response`, and passes them to a judge model (or `Ragas` evaluation pipeline) using a structured grading rubric:

* **Score 1 (Fail):** Contains unverified external facts or missing citations.

* **Score 2 (Partial):** Grounded in context, but misses key comparative nuances across papers.

* **Score 3 (Pass):** Completely grounded, synthesizes multi-paper evidence accurately, and attributes every claim to its exact source PDF.

## 5. Iterative Optimization Loop

If evaluation scores fall below target thresholds, apply systematic ablation adjustments:

* **Low** $\text{HR}@k$ **/** $\text{MRR}$**:** Adjust `chunk_size` (e.g., testing $800$ vs. $1200$ characters), increase `chunk_overlap`, or upgrade the embedding model in `ingest.py`.

* **Low Faithfulness:** Tighten negative constraints in `literature_agent/agent.py` or reduce temperature ($T = 0.0\text{ to }0.2$) to enforce deterministic extraction.