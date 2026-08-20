import arxiv
import os
from pathlib import Path
from urllib.request import urlretrieve
from collections.abc import Iterable
import pandas as pd
import pymupdf
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import json

from langchain_core.documents import Document

from ragstream.model_interaction import BooleanReasoning
from ragstream.grobid import GrobidExtractionError, extract_body_text as get_grobid_text
from ragstream.utils import imerge, cd

# logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def get_query_terms(json_filepath: str | Path = "research.json") -> Dict[str, Any]:
    """
    Retrieve the queries to search ArXiv from the research configuration.

    Args:
        json_filepath (str or Path): Path to the reasearch json configuration

    Raises:
        FileNotFoundError: Raised if the file specified does not exist.

    Returns: Dict[str, Any]
    """
    path = Path(json_filepath)
    if path.exists():
        with open(Path(json_filepath)) as f:
            research_json = json.loads(f.read())
        return research_json
    else:
        raise FileNotFoundError(f"Cannot find file at: {path}")

def download_pdf(pdf_url: str, name = "paper", save_dir: Path | str = Path(os.getcwd()) / "data") -> None:
    """
    Download a PDF to a specifies location.

    Args:
        pdf_url (str): The URL of the PDF. Duh.
        name (str): the name the PDF should be saved as.

    Returns:
        int: HTTP return code
    """
    if not name.endswith(".pdf"):
        name = name + ".pdf"
    with cd(save_dir):
        _, status = urlretrieve(pdf_url, name)
    if status != 200:
        logging.warning(f"HTTP retuned code {status}")
    return status

def get_arxiv_doi(pdf_url) -> str | None:
    """
    Extracts an ArXiv DOI from the URL

    Args:
        pdf_url (str): The URL of the PDF. Duh.

    Returns:
        str: DOI
    """
    if not pdf_url.startswith("https://arxiv.org/pdf/"):
        print(f"Invalid arXiv PDF URL: {pdf_url}")
        return None
    uid = pdf_url.split("/")[-1]
    return f"arXiv:{uid}"

def get_from_arxiv(queries: List[str] | str, max_results = 6, date_from: None = None, date_to: None = None) -> pd.DataFrame:
    """
    Search and retrieve papers from ArXiv servers with a list of queries

    Args:
        queries (list of strings, or string): The query or queries to seach ArXiv and get papers.
        max_results (int): The first n results to rerieve for each query parsed.
        date_from: TDB
        date_to: TDB

    Returns:
        pd.DataFrame: Entries of papers returned from ArXiv, with unique entries.
    """
    papers = pd.DataFrame(columns = ['entry_id', 'updated', 'published', 'title', 'authors', 'summary', 'comment', 'journal_ref', 'doi', 'primary_category', 'categories', 'links', 'pdf_url'])
    if not isinstance(queries, Iterable) or isinstance(queries, str):
        queries = [queries]
        assert len(queries) == 1
    client = arxiv.Client()
    for query in queries:
        logging.info(f"Searching for query: {query}")
        search = arxiv.Search(
            query = query,
            max_results = max_results,
            sort_by = arxiv.SortCriterion.SubmittedDate
        )
        results = client.results(search)
        for result in results:
            entry = result.__dict__

            # Hmmmmm...
            del entry['links']
            entry['authors'] = "; ".join([author.name for author in entry['authors']])
            entry['categories'] = "; ".join(entry['categories'])
            # End Hmmmmm...

            row = pd.DataFrame.from_dict(entry, orient = 'index').T
            papers = pd.concat([papers, row], ignore_index=True) # absolutely disgusting, but since .append() was dropped... yeah. Poor decision.
    papers = papers.drop_duplicates(subset=['entry_id'])
    logging.info(f"Created DataFrame with {len(papers)} entries.")
    papers['doi'] = papers['pdf_url'].apply(get_arxiv_doi)
    return papers

def extract_article_text(pdf_path: str | Path) -> List[Document] | str:
    """
    Extract text from a given PDF. Primarily, this method uses GROBID, but if that fails, the 
    method falls back to a slightly more improcise approach using PyMuPDF which does a poorer
    job of stripping whitespace and various other document sections.

    Args:
        pdf_path (Path or str): The path of the PDF to be extracted.

    Returns:
        str: Extracted text.
    """
    text = None
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        text = get_grobid_text(pdf_path)
    except GrobidExtractionError:
        logging.warning("GROBID failed. Falling back to PyMuPDF. Expect drop in extraction quality.")
        try:
            with pymupdf.open(pdf_path) as doc:
                text = chr(12).join(
                    [page.get_text(sort = True) for page in doc]
                )
        except Exception as e:
            logging.error(f"Error occurred while reading PDF at {pdf_path}: {e}")
    return text

def get_latest_research(
        topic: str,
        queries: str | List[str],
        n_recent: int = 5,
        as_langchain_document: bool = True
        ) -> List[Tuple[str, str]]:
    """
    Queries ArXiv and uses LLM reasoning functionality, finding relevent papers from recent research.

    Args:
        topic (str): The research topic against which relevence is determined
        queries (List of str): queries for getting papers from ArXiv
        n_recent (int): Number of recent papers to retrieve for each query. Defaults to 5.
        as_langchain_document (bool): Whether or not a list of langchain docs or just as list of article text

    Returns:
        List of Tuple[str, str]: The ArXiv DOI and extracted text from each PDF found to be relevent to the research topic.
    """
    papers_df = get_from_arxiv(
        queries = queries,
        max_results = n_recent
    )
    latest = []
    summaries = papers_df["summary"].tolist()
    urls = papers_df["pdf_url"].tolist()
    dois = papers_df["doi"].tolist()
    decisions = np.zeros(len(summaries), dtype=np.int8)
    for i, summary in enumerate(summaries):
        # New instance each loop to avoid adding to context and creating model confusion etc.
        decision_maker = BooleanReasoning(additional_instructions = f"Would this paper likely aid in research on the topic: {topic}")
        _, decision = decision_maker.invoke(summary)
        logging.info(f"Decision on {dois[i]}: {decision}")
        doi = dois[i]
        if decision == "yes":
            decisions[i] = 1
            name = doi.replace(":", "_")
            download_pdf(urls[i], name = name)
            pages = extract_article_text(Path(os.getcwd()) / "data" / f"{name}.pdf")
            if as_langchain_document:
                for j, page in enumerate(pages):
                    document = Document(
                        page_content=page,
                        metadata={
                            "page" : j
                            "source": urls[i],
                            "DOI": doi
                        }
                    )
                    latest.append(document)
            else:
                latest.append(pages)
    decisions = np.bool(decisions)
    if np.all(~decisions):
        logging.warning("No relevant papers found. Refine your queries or topic.")
        return []
    else:
        logging.info(f"Fetched {len(latest)} relevant papers.")
        return latest

if __name__ == "__main__":
    import json

    with open(Path(os.getcwd()) / "research.json") as f:
        research_json = json.loads(f.read())
    print(research_json)
    topic = research_json["topic"]
    queries = research_json["queries"]

    df = get_from_arxiv(
        queries,
        max_results = 10
    )
    summaries = df["summary"].tolist()
    urls, dois = df["pdf_url"].tolist(), df["doi"].tolist()
    for i, summary in enumerate(summaries[2:5]):  # Changed to iterate over a slice of the summaries
        # summariser = Summarisation(additional_instructions = "Please provide a concise summary in 1-2 sentences.")
        decision_maker = BooleanReasoning(additional_instructions = f"Would this paper likely aid in some {topic}")
        # summary_output = summariser.invoke(summary)
        _, decision = decision_maker.invoke(summary)    
        # print(f"Model Summary: {summary_output}")
        print(f"Decision: +++ {decision} +++")
        if decision == "yes":
            name = dois[i].replace(":", "_")
            download_pdf(urls[i], name = name)
            print(f"Downloaded PDF for paper {i+1} to data/{name}.pdf")
            text = extract_article_text(Path(os.getcwd()) / "data" / f"{name}.pdf")
            print(f"Extracted text from PDF for paper {i+1}:\n{text[:500]}...") # Print first 500 characters of the extracted text
