"""
Example `research.json`:
```
{
  "topic": "structure discovery pipeline for novel battery materials",
  "queries": [
    "'Sodium ion battery' AND 'condensed matter'",
    "NASICON AND electrode",
    "'Sodium battery' AND 'novel materials'"
  ]
}
```
"""

from ragstream.papers import get_query_terms, get_latest_research
from ragstream.rag import RAGPipline

# 1. Optionally extract research topic and search queries from a JSON file
research_area = get_query_terms("research.json")

# 2. Apply topic and search queries, then fetch relevent research papers and text
relevant_papers = get_latest_research(
    topic=research_area["topic"],
    queries=research_area["queries"],
    n_recent=25,
)

# 3. RAGify the relevant research

rp = RAGPipline(db_name = "./chromastore.db")
rp.add_documents(relevant_papers)
print(f"RAG pipeline created with {len(relevant_papers)} documents and stored in {rp.db_name}")

# 4. Query the specific knowledgebase