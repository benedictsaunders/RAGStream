import os
from typing import List, Tuple, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dataclasses import dataclass, field
import re
import logging

from ragstream.model_interaction import GroundedAnswer, QueryDecomposition

logger = logging.getLogger(__name__)

@dataclass
class Citation:
    """A source that the answer actually cited."""
    number: int
    doi: str
    title: str
    page: int
    source: str

@dataclass
class RAGAnswer:
    """
    The result of a query. `retrieved` holds every chunk given to the model with its cross-encoder score (higher
    is more relevant), whether or not it was cited.
    """
    question: str
    answer: str
    sub_questions: List[str] = field(default_factory = list)
    citations: List[Citation] = field(default_factory = list)
    retrieved: List[Tuple[Document, float]] = field(default_factory = list)
    invalid_citations: List[int] = field(default_factory = list)
    abstained: bool = False

    def __str__(self) -> str:
        references = "\n".join(
            f"[{c.number}] {c.title} ({c.doi}, p.{c.page}) {c.source}" for c in self.citations
        )
        return f"{self.answer}\n\n{references}" if references else self.answer

class RAGPipline:
    """
    Docs
    """
    def __init__(self, db_name: str = "./chromastore.db", collection_name = "papers") -> None:
        """

        """
        self.db_name = db_name

        self._embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._textsplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index = True,
        )
        self._vectorstore = Chroma(
            collection_name = collection_name,
            embedding_function = self._embeddings,
            persist_directory=db_name
        )

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents as Langchain Documents

        Args:
            documents (list of Langchain Documents): The documents to be ragged

        Returns:
            int: the number of chunks added to database
        """
        if not all([isinstance(doc, Document) for doc in documents]):
            raise TypeError("Expeced list of LangChain documents ('add_documents')")
        chunks = self._textsplitter.split_documents(documents)
        if not chunks:
            return 0
        # Deterministic IDs, so re-adding the same chunk overwrites rather than duplicates
        doc_ids = [
            f"{chunk.metadata['DOI']}:{chunk.metadata['section']}:p{chunk.metadata['page']}:{chunk.metadata['start_index']}"
            for chunk in chunks
        ]
        self._vectorstore.add_documents(chunks, ids = doc_ids)
        return len(chunks)

    def count_chunks(self) -> int:
        """
        Returns:
            int: the number of chunks in the database
        """
        return len(self._vectorstore.get(include = [])["ids"])

    def retrieve(
            self,
            question: str,
            k: int = 6,
            fetch_k: int = 50,
            min_score: float | None = None
        ) -> List[Tuple[Document, float]]:
        """
        Find the chunks most relevant to the question. The closest `fetch_k` chunks by embedding are re-scored
        against the question by a cross-encoder, and the best `k` of those are kept. The embedding search alone
        finds the right paper but often ranks the chunk holding the answer too low to be kept.

        Args:
            question (str): The question to search with.
            k (int): Number of chunks to return. Defaults to 6.
            fetch_k (int): Number of closest chunks to re-score. Defaults to 50.
            min_score (float, optional): Drop chunks the cross-encoder scores below this. Defaults to None (keep all).

        Returns:
            List of Tuple[Document, float]: Chunks and their cross-encoder scores (higher is more relevant), best first.
        """
        candidates = [doc for doc, _ in self._vectorstore.similarity_search_with_score(question, k = max(k, fetch_k))]
        if not candidates:
            return []
        scores = self._reranker.predict(
            [(question, doc.page_content) for doc in candidates],
            show_progress_bar = False
        )
        hits = sorted(zip(candidates, [float(score) for score in scores]), key = lambda hit: hit[1], reverse = True)
        if min_score is not None:
            hits = [(doc, score) for doc, score in hits if score >= min_score]
        return hits[:k]

    def query(
            self,
            question: str,
            k: int = 6,
            fetch_k: int = 50,
            min_score: float | None = None,
            decompose: bool = True
        ) -> RAGAnswer:
        """
        Answer a question from the stored papers, with citations.

        Args:
            question (str): The question to be answered.
            k (int): Number of chunks to give the model. Defaults to 6.
            fetch_k (int): Number of closest chunks to re-score for each search. Defaults to 50.
            min_score (float, optional): Drop chunks the cross-encoder scores below this. Defaults to None (keep all).
            decompose (bool): Whether to split the question into separate searches first, which costs one extra
                LLM call. Without it, a question covering two topics only retrieves chunks for one. Defaults to True.

        Returns:
            RAGAnswer: The answer, the sources it cited, and everything that was retrieved.
        """
        sub_questions = QueryDecomposition().invoke(question) if decompose else [question]
        ranked = [self.retrieve(sq, k = k, fetch_k = fetch_k, min_score = min_score) for sq in sub_questions]

        # Take the best chunk of each search in turn, so that every sub-question gets a share of the k slots
        hits, seen = [], set()
        for position in range(k):
            for results in ranked:
                if position >= len(results) or len(hits) >= k:
                    continue
                doc, score = results[position]
                key = (doc.metadata["DOI"], doc.metadata["section"], doc.metadata["page"], doc.metadata["start_index"])
                if key not in seen:
                    seen.add(key)
                    hits.append((doc, score))

        if not hits:
            logger.warning("Nothing relevant retrieved. Skipping the LLM.")
            return RAGAnswer(
                question = question,
                answer = "No relevant sources were found for this question.",
                sub_questions = sub_questions,
                abstained = True
            )

        # The model cites by number, and the numbers are mapped back to the metadata below
        context = "\n\n".join(
            f"[{n}] {doc.metadata['title']} ({doc.metadata['DOI']}, p.{doc.metadata['page']})\n{doc.page_content}"
            for n, (doc, _) in enumerate(hits, start = 1)
        )
        answer = GroundedAnswer().invoke(question, context)
        # Some models seem to cite with full-width brackets, i.e.【1】. Annoying.
        answer = answer.replace("【", "[").replace("】", "]")

        # Matches [1] as well as [1, 3]
        cited = []
        for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
            for number in group.split(","):
                number = int(number)
                if number not in cited:
                    cited.append(number)

        invalid = [n for n in cited if not 1 <= n <= len(hits)]
        if invalid:
            logger.warning(f"Answer cited sources that were not provided: {invalid}")

        citations = []
        for n in sorted(set(cited) - set(invalid)):
            metadata = hits[n - 1][0].metadata
            citations.append(Citation(
                number = n,
                doi = metadata["DOI"],
                title = metadata["title"],
                page = metadata["page"],
                source = metadata["source"]
            ))

        return RAGAnswer(
            question = question,
            answer = answer,
            sub_questions = sub_questions,
            citations = citations,
            retrieved = hits,
            invalid_citations = invalid,
            abstained = not citations
        )

