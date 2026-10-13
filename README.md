# RAGStream

A RAG pipeline designed specifically for searching for and ingesting scientific publications en masse. Currently, this is limited to arXiv, but who knows where else it'll scrape! It also condenses the article summaries and queries whether a given paper is suitable literature for a given research project.

I have avoided using LLMs to write the code in this repository.

### Yet another RAG pipeline???

This was designed initially with a specific overarching project in mind, using a local LLM to assist in structure search pipelines. However, it can be used to keep up to date on any research topic on arXiv, with the `research.json` file. For example, looking at papers that could help to inform on structures and composition for novel Li-ion battery material discovery:
```json

{
  "topic": "structure discovery pipeline for novel battery materials",
  "queries": [
    "'lithium ion battery' AND 'condensed matter'",
    "'LISICON' AND 'electrode'",
    "'lithium battery' AND 'solid electrolyte'",
    "'lithium battery' AND 'novel materials'",
    "'structure discovery' AND 'battery materials'",
    "'lithium battery' AND 'high throughput'",
    "'lithium battery' AND 'machine learning'",
    "'lithium battery' AND 'data mining'",
    "'lithium battery' AND 'computational chemistry'",
    "'lithium battery' AND 'materials informatics'"
  ]
}
```
You can have as many queries as you like, using standard boolean operators in the search terms, as you would when searching on arXiv yourself. This can be used in the workflow outlined in `example.py`, rather than the CLI tool.

And also, imagine using a tool called YARG. Hey, have you seen the latest YARG update? 🤮

## Requirements
At the moment, it is probably easiest to use `uv init --bare && uv sync` to get this project going, after cloning or downloading the repo. You will also require an LLM for this, naturally. I use a local model, running with `llama.cpp` at `127.0.0.1:8080`.
Other than what is stated in `pyproject.toml` and the LLM, this project also requires GROBID (), a local instance of which can be acquired and run using a Docker image:
```
docker run --rm --init --ulimit core=1 -p 8070:8070 grobid/grobid:0.9.0-crf
```
Currently, I use the following bash script to launch both `GROBID` and `llama-server` simultaneously, printing the Docker container ID and the `PID` of the llama instance:
```bash
#!/bin/bash

# Start GROBID

sudo docker run --rm --init --ulimit core=1 -p 8070:8070 grobid/grobid:0.9.0-crf > grobid.log 2>&1 &

# Start LLM

[llama server command] > llama.log 2>&1 &
sleep 1

# Get PID and GROBID container ID

GID=$(sudo docker ps | grep grobid | awk '{print $1}')
echo "LLM PID: $!"
echo "GROBID CONTAINER ID: $GID"
```
Make sure that GROBID and Llama use different ports!

## Quickstart

For a search of arXiv using the `research.json` schema illustrated above, and the CLI tool, once the `uv` environment is set up, is as easy as
```bash
ragstream ask "What is a question?"
```
which asks a question on the vector database built using `example.py`. Or when you get stuck, you can always ask for help:
```bash
ragstream help
```

## Progress
| Area | Status |
|---|---|
| Pipeline (search, filter, extract, ingest, query) | Working e2e |
| Evaluation | Working on it now! |
| Retrieval quality | Partly done |
| Tracking LLM use | Not started |
| Bugs | 3 of 5 fixed |

### Pipeline
- [x] Paper querying
- [x] Sorting, summary, and download
- [x] Text extraction
- [x] Ingestion into a persistent vector store, safe to re-run
- [x] Querying with cited answers, and abstaining when the papers do not contain the answer

### Evaluation
- [ ] Retrieval: 30-50 hand-written questions, each with the paper and page that answers it, reporting recall@k and MRR
- [ ] Relevance filter: precision and recall of the yes/no decision against hand-labelled abstracts
- [ ] Ablation table: chunk size, embedding model, with and without the reranker

### Retrieval quality
- [x] Cross-encoder reranker
- [x] Abstracts ingested as their own chunks
- [x] Splitting multi-topic questions into separate searches
- [x] Title and section in the chunk metadata
- [ ] Authors, date, and category in the chunk metadata
- [ ] Section-aware chunking using the GROBID structure
- [ ] An embedding model better suited to scientific text
- [ ] Hybrid BM25 and dense retrieval
- [ ] Reliable page numbers (GROBID paragraphs without location currently default to page 1)

### LLM Usage
- [ ] Latency, tokens, and VRAM reported per stage
- [ ] Incremental runs that only pick up papers published since the last run

### Known bugs
- [x] Relevance decisions crashed for more than one paper
- [x] The PyMuPDF fallback produced one document per character
- [x] A download warning is logged even when the download succeeds
- [ ] The retry while the LLM is still loading (HTTP 503) is never reached
- [ ] Importing the package raises if the services are not running

### More feaures
- [ ] Use a decision model, like Jev, instead or using the primary LLM as a judge when reading summaries/abstracts

### ???
- [ ] The future...

## Philosophy
The principal idea behind this project is to ascertain the state of the art for a very specific, very niche research topic. Below is a short workflow concept, but the actual order of operations will eventually be able to be user-defined for various applications.

1. Enter queries for arXiv papers
2. Use an LLM to ascertain whether the paper might be worth downloading, using the summary as retrieved from step 1.
3. Download and extract the text from the papers.
4. Apply RAG methodologies to query these papers on the state of the art and other bleeding-edge research topics and questions.
