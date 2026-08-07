import arxiv
import os
from pathlib import Path
from urllib.request import urlretrieve
from collections.abc import Iterable
from utils import imerge, cd
import pandas as pd
from pprint import pprint
import pymupdf
from model_interaction import BooleanReasoning, Summarisation
from collections import Counter

def download_pdf(pdf_url, name = "paper", save_dir = Path(os.getcwd()) / "data"):
    with cd(save_dir):
        urlretrieve(pdf_url, f"{name}.pdf")

def get_arxiv_doi(pdf_url):
    if not pdf_url.startswith("https://arxiv.org/pdf/"):
        print(f"Invalid arXiv PDF URL: {pdf_url}")
        return None
    uid = pdf_url.split("/")[-1]
    return f"arXiv:{uid}"

def get_from_arxiv(queries, max_results = 6, date_from = None, date_to = None) -> pd.DataFrame:
    papers = pd.DataFrame(columns = ['entry_id', 'updated', 'published', 'title', 'authors', 'summary', 'comment', 'journal_ref', 'doi', 'primary_category', 'categories', 'links', 'pdf_url'])
    if not isinstance(queries, Iterable) or isinstance(queries, str):
        queries = [queries]
        assert len(queries) == 1
    client = arxiv.Client()
    for query in queries:
        print(f"Searching for query: {query}")
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
    papers['doi'] = papers['pdf_url'].apply(get_arxiv_doi)
    return papers

def get_article_text(pdf_path):
    pdf_path = Path(os.getcwd()) / Path(pdf_path)
    print(f"Extracting text from PDF at {pdf_path}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        with pymupdf.open(pdf_path) as doc:
            print(doc.__dict__.keys())
            text = chr(12).join(
                [page.get_text(sort = True) for page in doc]
            )
    except Exception as e:
        print(f"Error occurred while reading PDF at {pdf_path}: {e}")
    return text

def extract_body_text(pdf_path):
    doc = pymupdf.open(pdf_path)
    font_sizes = []
    text_blocks = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        font_sizes.append(round(s["size"]))
                        text_blocks.append({
                            "text": s["text"],
                            "size": round(s["size"]),
                            "bbox": b["bbox"] # [x0, y0, x1, y1]
                        })
    if not font_sizes:
        return ""
    body_size = Counter(font_sizes).most_common(1)[0][0]

    body_text = []
    for block in text_blocks:
        if block["size"] == body_size:
            y0 = block["bbox"][1]
            if 70 < y0 < 720: 
                body_text.append(block["text"])

    return " ".join(body_text)

if __name__ == "__main__":
    pdf = Path(os.getcwd()) / "data" / "arXiv_2608.04856v1.pdf"
    text = extract_body_text(pdf)
    print(f"Extracted text from PDF:\n{text[:500]}...")  # Print first 500 characters of the extracted text

if __name__ == "__main__":
    df = get_from_arxiv(
        "<SOME QUERY>",
        max_results = 10
    )
    summaries = df["summary"].tolist()
    urls, dois = df["pdf_url"].tolist(), df["doi"].tolist()
    for i, summary in enumerate(summaries[2:5]):  # Changed to iterate over a slice of the summaries
        summariser = Summarisation(additional_instructions = "Please provide a concise summary in 1-2 sentences.")
        decision_maker = BooleanReasoning(additional_instructions = "Would this paper likely aid in <RESEARCH DOMAIN> research?")
        # summary_output = summariser.invoke(summary)
        _, decision = decision_maker.invoke(summary)    
        # print(f"Model Summary: {summary_output}")
        print(f"Decision: +++ {decision} +++")
        if decision == "yes":
            name = dois[i].replace(":", "_")
            download_pdf(urls[i], name = name)
            print(f"Downloaded PDF for paper {i+1} to data/{name}.pdf")
            text = get_article_text(f"data/{name}.pdf")
            print(f"Extracted text from PDF for paper {i+1}:\n{text[:500]}...")  # Print first 500 characters of the extracted text
