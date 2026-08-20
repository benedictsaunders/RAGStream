import logging
from lxml import etree
from pathlib import Path
import re
from grobid_client.grobid_client import GrobidClient
from typing import List

logger = logging.getLogger(__name__)

class GrobidExtractionError(Exception):
    """Custom exception for GROBID extraction errors."""
    pass

def extract_body_text(pdf: Path | str,
                       keep_pages: bool = True,
                       server_url: str = "http://localhost:8070"
                       ) -> str | List[str]:
    """
    Extracts the body text from a PDF using GROBID.

    Args:
        pdf (Path or str): Path to the PDF file.
        keep_pages (bool): Whether or not to return a list of pages, or a single block, defaults to True
        server_url (str, optional): URL of the GROBID server. Defaults to 'http://localhost:8070'.

    Returns:
        str: Extracted body text from the PDF.
    Raises:
        GrobidExtractionError: If the GROBID request fails or if the body text cannot be read/found/identified.
    """
    TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
    client = GrobidClient(grobid_server=server_url)
    try:
        _, status, tei_xml = client.process_pdf(
            "processFulltextDocument",
            str(pdf),
            generate_ids=False,
            consolidate_header=True,     # cleans up title/author metadata (cheap, local)
            consolidate_citations=False, # skip CrossRef lookups -- not needed for body text
            include_raw_citations=False,
            include_raw_affiliations=False,
            tei_coordinates=True,
            segment_sentences=False,
        )
    except Exception as e:
        raise GrobidExtractionError(f"GROBID request failed: {e}")

    if status != 200:
        raise GrobidExtractionError(f"GROBID request failed with status code {status}")

    try:
        root = etree.fromstring(tei_xml.encode("utf-8"))
    except Exception as e:
        raise GrobidExtractionError(f"Failed to parse GROBID XML: {e}")

    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is None:
        logging.warning("No body text found in GROBID output.")
        return [] if keep_pages else ""

    citation_re = re.compile(r"\[\d+(?:,\s*\d+)*\]")

    pages_dict: dict[int, list[str]] = {}
    paragraphs: list[str] = []

    for pi, p in enumerate(body.findall(".//tei:p", TEI_NS)):
        text = "".join(p.itertext()).strip()
        text = " ".join(text.split())
        if not text:
            continue
        text = citation_re.sub("", text).strip()
        if not text:
            logging.warning(f"No text found for page {pi}.")
            continue

        if not keep_pages:
            paragraphs.append(text.strip())
            continue

        coords = p.get("coords")
        if coords:
            start_page = int(coords.split(",")[0])
        else:
            logging.warning("Paragraph missing coords; defaulting to page 1. "
                             "Check that tei_coordinates covers <p> elements.")
            start_page = 1

        pages_dict.setdefault(start_page, []).append(text)

    if not keep_pages:
        return "\n\n".join(paragraphs)

    if not pages_dict:
        return []

    max_page = max(pages_dict.keys())
    pages_list = [
        "\n\n".join(pages_dict.get(i, []))
        for i in range(1, max_page + 1)
    ]
    return pages_list