# RAGStream

This is (going to be) a RAG pipeline designed specifically for searching for and ingesting scientific publications en masse. Currently, this is limited to ArXiv, but who knows where else it'll scrape! It will also condense the article summary, and query whether a given paper is suitable literature for a research project.

I have avoided using LLMs to write the code in this repository.

### Yet another RAG pipeline???

Well, this is designed with a specific project in mind (watch this space!) using a local LLM, which adds a further challenge of resource constraints. And also YARG isn't a particularly nice-sounding project name.

## Progress
- [x] Paper querying
- [x] Sorting, summary, and download
- [x] Text extraction
- [ ] RAG implementation
- [ ] The future???

## Requirements
At the moment, it is probably easiest to use `uv init --bare && uv sync` to get this project going, after cloining or downloading the repo. You will also recquire an LLM for this, naturally. I use a local model, running with `llama.cpp` at `127.0.0.1:8080`.
Other than what is stated in `pyproject.toml` and the LLM, this project also requires GROBID (), a local instant of which can be acquired and run using a docker image (taken from ):
```
docker run --rm --init --ulimit core=1 -p 8070:8070 grobid/grobid:0.9.0-crf
```
Currently, I use the following bash script to launch both `GROBID` and `llama-server` simultaneously, printing the docker container ID and the `PID` of the llama instance:
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

## Philosophy
The principle idea behind this project is to ascertain the state of the at for a very specific, very niche research topic. Below is a short workflow concept, but the actual order of operations will eventually be able to be user-defined for various applications.

1. Enter queries for ArXiv papers
2. Use an LLM to ascertain whether the paper might be worth downloading, using the summary as retrieved from step 1.
3. Download and extract the text from the papers.
4. Apply RAG methodologies to query from these papers as to the state of the art, and other bleeding-edge research topics and questions.
