from ragstream.papers import get_query_terms, get_latest_research

# 1. Optionally extract research topic and search queries from a JSON file
research_area = get_query_terms("research.json")

# 2. Apply topic and search queries, then fetch relevent research papers and text
relevant_papers = get_latest_research(
    topic=research_area["topic"],
    queries=research_area["queries"]
)

# 3. RAGify the relevant research

# 4. Query the specific knowledgebase