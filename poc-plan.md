# PoC Plan: Kwipu

## Project Classification
- **Type:** rag
- **Key Technologies:** LlamaIndex, Ollama, Property Graph Index, Hybrid Retrieval (Vector + BM25 + Temporal)
- **ODH Relevance:** Demonstrates RAG capabilities with local LLM inference via Ollama, aligns with ODH's model serving and knowledge graph use cases

## PoC Objectives
What we want to prove:
1. Kwipu can build a knowledge graph from markdown notes using LLM-based entity extraction
2. The system successfully executes hybrid retrieval queries across multiple documents
3. Multilingual support works with Italian/English document processing
4. Real-time synchronization detects changes in the knowledge base directory

## Infrastructure Requirements
- **Inference Server:** Ollama (sidecar container)
- **Vector Database:** in-memory (LlamaIndex)
- **Embedding Model:** nomic-embed-text (via Ollama)
- **GPU Required:** no
- **Persistent Storage:** none
- **Resource Profile:** large (for LLM processing)
- **Sidecar Containers:** ollama
- **Extra Environment Variables:** OLLAMA_HOST (required)

## Test Scenarios
### Scenario 1: graph-initialization
- **Description:** Verify the system builds the knowledge graph from example markdown files
- **Type:** cli
- **Input:** `python geode_graph.py --fast`
- **Expected:** Job completes with graph index created in storage_graph/ directory
- **Timeout:** 300 seconds

### Scenario 2: query-validation
- **Description:** Test query execution against the knowledge graph
- **Type:** cli
- **Input:** `python geode_graph.py --fast -q 'What is Project Alpha?'`
- **Expected:** Job returns a response containing information from Project Alpha.md
- **Timeout:** 120 seconds

### Scenario 3: multilingual-support
- **Description:** Verify multilingual processing capabilities
- **Type:** cli
- **Input:** `python geode_graph.py --fast -l it -q 'Cos'è il progetto Alpha?'`
- **Expected:** Job returns a response in Italian containing information from Project Alpha.md
- **Timeout:** 120 seconds

## Dockerfile Considerations
This is a CLI-based application that builds a knowledge graph and runs queries. The container should:
- Install requirements.txt dependencies
- Set entry point to `python geode_graph.py`
- No port exposure needed
- Include OLLAMA_HOST environment variable for Ollama communication
- Example: "This is a CLI tool. ENTRYPOINT should be the Python script. CMD should default to --help. Do NOT add EXPOSE — there is no port to expose."

## Deployment Considerations
Deploy as a Kubernetes Job that runs the CLI commands. No Service needed since there's no port. Testing via kubectl run --rm with specific command arguments. The Ollama service must be available as a sidecar container for LLM inference. Example: "Deploy as a Job with specific command arguments. Do NOT create a Service — there is no port. Test via kubectl run --rm with query parameters. Require Ollama as a sidecar container for LLM inference."