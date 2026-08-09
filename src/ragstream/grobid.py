import logging
from lxml import etree
from pathlib import Path
import re
from grobid_client.grobid_client import GrobidClient

logger = logging.getLogger(__name__)

class GrobidExtractionError(Exception):
    """Custom exception for GROBID extraction errors."""
    pass

def extract_body_text(pdf: Path | str, server_url: str = "http://localhost:8070") -> str:
    """
    Extracts the body text from a PDF using GROBID.

    Args:
        pdf (Path or str): Path to the PDF file.
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
            tei_coordinates=False,
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
        return ""

    paragraphs = []
    for p in body.findall(".//tei:p", TEI_NS):
        text = "".join(p.itertext()).strip()
        text = " ".join(text.split())
        if text:
            paragraphs.append(text)

    body = "\n\n".join(paragraphs)
    body = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", body)
    return body