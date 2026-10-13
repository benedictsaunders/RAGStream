import argparse
import logging
import readline # Gives input() line editing and history with the arrow keys
import sys
from pathlib import Path

from ragstream.utils import check_services

def open_pipeline(db_name: str):
    """
    Open an existing store for querying. Nothing is searched for, downloaded or added, so the same questions
    keep being asked of the same papers. Only the LLM is needed, not GROBID.

    Args:
        db_name (str): Path of the store to open.

    Returns:
        RAGPipline: The pipeline, with the embedding model and reranker loaded.
    """
    if not Path(db_name).exists():
        sys.exit(f"No store found at {db_name}. Ingest some papers first.")
    if not check_services(ports = {"LLM": 8080}):
        sys.exit("The LLM (llama.cpp) is not running on port 8080.")

    # Imported here as it is slow, and pointless if the checks above fail
    from ragstream.rag import RAGPipline
    rp = RAGPipline(db_name = db_name)
    if rp.count_chunks() == 0:
        sys.exit(f"The store at {db_name} is empty. Ingest some papers first. Either use RAGStream")
    return rp

def ask(args: argparse.Namespace) -> None:
    """Answer a single question and exit.
    
    Args:
        args (argparse.Namespace):

    """
    rp = open_pipeline(args.db)
    print(rp.query(args.question, k = args.k, decompose = not args.no_decompose))

def chat(args: argparse.Namespace) -> None:
    """
    Answer questions until told to stop. The models are loaded once, rather than once per question. Each question
    is answered on its own, with no memory of the previous ones.

    Args:
        args (argparse.Namespace):

    """
    rp = open_pipeline(args.db)
    print(f"\n{rp.count_chunks()} chunks in {args.db}. Ask a question, or 'exit' to leave.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        try:
            answer = rp.query(question, k = args.k, decompose = not args.no_decompose)
        except Exception as e:
            # Losing the LLM for one question should not end the session
            print(f"Query failed: {e}")
            continue
        print(f"\n{answer}")

def clear(args: argparse.Namespace) -> None:
    """
    Delete every chunk in the store, after asking for confirmation. The store itself is kept, and is filled again
    by the next ingestion. The downloaded PDFs are left alone, and neither the LLM nor GROBID is needed.

    Args:
        args (argparse.Namespace):

    """
    # Checked first, as opening a path with chromadb creates a store there
    if not (Path(args.db) / "chroma.sqlite3").exists():
        sys.exit(f"No store found at {args.db}. Nothing to clear.")

    # chromadb directly rather than RAGPipline, which would load the embedding model and reranker just to delete
    import chromadb
    client = chromadb.PersistentClient(path = args.db)
    COLLECTION = "papers"
    try:
        n_chunks = client.get_collection(COLLECTION).count()
    except chromadb.errors.NotFoundError:
        n_chunks = 0
    if n_chunks == 0:
        print(f"The store at {args.db} is already empty.")
        return

    if not args.yes:
        try:
            reply = input(f"Delete all {n_chunks} chunks in {args.db}? This cannot be undone. [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            reply = ""
        if reply.lower() not in ("y", "yes"):
            print("Nothing was deleted.")
            return

    client.delete_collection(COLLECTION)
    print(f"Deleted {n_chunks} chunks from {args.db}.")

    if args.pdfs:
        # The same directory that download_pdf saves to
        for pdf in Path("data").glob("*.pdf"):
            pdf.unlink()
        print("Deleted the PDFs in ./data.")

def reconstruct(args: argparse.Namespace) -> None:
    """
    Rebuild the store from the PDFs already in ./data, without searching for or downloading anything. The titles
    and abstracts are not in the extracted text, so they are looked up on ArXiv from the file names. Only GROBID
    is needed, not the LLM. Chunks already in the store are overwritten rather than duplicated.

    Args:
        args (argparse.Namespace):

    """
    # The same directory and naming that download_pdf saves with
    pdfs = sorted(Path("data").glob("arXiv_*.pdf"))
    if not pdfs:
        sys.exit("No ArXiv PDFs found in ./data. Nothing to reconstruct from.")
    if not check_services(ports = {"GROBID": 8070}):
        sys.exit("GROBID is not running on port 8070.")

    # Imported here as they are slow, and pointless if the checks above fail
    import arxiv
    from ragstream.papers import create_documents, extract_article_text
    from ragstream.rag import RAGPipline

    ids = [pdf.stem.removeprefix("arXiv_") for pdf in pdfs]
    papers = {result.get_short_id(): result for result in arxiv.Client().results(arxiv.Search(id_list = ids))}

    rp = RAGPipline(db_name = args.db)
    for pdf, arxiv_id in zip(pdfs, ids):
        paper = papers.get(arxiv_id)
        pages = extract_article_text(pdf) if paper else None
        if not pages:
            print(f"{pdf.name}: skipped, {'no text extracted' if paper else 'not found on ArXiv'}.")
            continue
        documents = create_documents(pages, paper.summary, url = paper.pdf_url, doi = f"arXiv:{arxiv_id}", title = paper.title)
        print(f"{pdf.name}: {rp.add_documents(documents)} chunks.")
    print(f"\n{rp.count_chunks()} chunks in {args.db}.")

def help(args: argparse.Namespace) -> None:
    """
    Print the help pane/paragraphs and exit

    Args:
        args (argparse.Namespace):

    """
    print(
        """
        RAGStream answers questions from a store of ingested scientific papers, citing the papers it used.

        Asking questions requires the LLM (llama.cpp) running on port 8080 by default, and a store that already holds some papers.
        GROBID is only needed for ingesting, not for asking. By default, GROBID is set to be running on port 8070.

        ragstream chat                     Ask questions one after another. The models are only loaded once.
        ragstream ask "What is ...?"       Ask a single question and exit.
        ragstream clear                    Delete everything in the store, after asking first (-y to skip).
        ragstream reconstruct              Rebuild the store from the PDFs in ./data. Needs GROBID, but not the LLM.
        ragstream help                     Show this help message.

        The commands chat, ask, clear, and reconstruct take --db to point at a store other than the default ./chromastore.db. Each question is answered on
        its own, with no memory of the previous ones. If an answer misses part of a question, try a larger -k. If the
        searches drift away from what was asked, try --no-decompose. `ragstream <command> -h` lists the options of
        a command.
        """
    )

def main() -> None:
    parser = argparse.ArgumentParser(prog = "ragstream", description = "Ask questions of ingested papers.")
    
    # --db is kept apart from the query options, as clear needs the store but not -k or --no-decompose. This is getting messy...
    store = argparse.ArgumentParser(add_help = False)
    store.add_argument("--db", default = "./chromastore.db", help = "Path of the store. Defaults to ./chromastore.db")
    
    options = argparse.ArgumentParser(add_help = False)
    options.add_argument("-k", type = int, default = 6, help = "Number of chunks given to the LLM. Defaults to 6.")
    options.add_argument("--no-decompose", action = "store_true", help = "Do not split the question into separate searches.")

    subparsers = parser.add_subparsers(required = True)

    chat_parser = subparsers.add_parser("chat", parents = [store, options], help = "Ask questions interactively.")
    chat_parser.set_defaults(func = chat)

    ask_parser = subparsers.add_parser("ask", parents = [store, options], help = "Ask a single question.")
    ask_parser.add_argument("question")
    ask_parser.set_defaults(func = ask)

    clear_parser = subparsers.add_parser("clear", parents = [store], help = "Delete everything in the store.")
    clear_parser.add_argument("-y", "--yes", action = "store_true", help = "Do not ask for confirmation.")
    clear_parser.add_argument("--pdfs", action = "store_true", help = "Also delete the downloaded PDFs in ./data.")
    clear_parser.set_defaults(func = clear)

    reconstruct_parser = subparsers.add_parser("reconstruct", parents = [store], help = "Rebuild the store from the PDFs in ./data.")
    reconstruct_parser.set_defaults(func = reconstruct)

    help_parser = subparsers.add_parser("help", help = "Show help.")
    help_parser.set_defaults(func = help)

    args = parser.parse_args()

    # Otherwise every request to the LLM is logged in between the answers
    for noisy in ("httpx", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)
