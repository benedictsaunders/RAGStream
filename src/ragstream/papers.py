import arxiv
import os
from pathlib import Path
from urllib.request import urlretrieve
from collections.abc import Iterable
import pandas as pd
import pymupdf
import logging
from typing import List

from ragstream.model_interaction import BooleanReasoning, Summarisation
from ragstream.grobid import GrobidExtractionError, extract_body_text as get_grobid_text
from ragstream.utils import imerge, cd

# logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

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
        # if len(results) == 0:
        #     logging.warning(f"Query \"{query}\" returned 0 results.")
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

def extract_article_text(pdf_path: str | Path) -> str:
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
        summariser = Summarisation(additional_instructions = "Please provide a concise summary in 1-2 sentences.")
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
            print(f"Extracted text from PDF for paper {i+1}:\n{text[:500]}...")  # Print first 500 characters of the extracted text
