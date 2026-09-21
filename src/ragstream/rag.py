import os
from typing import List, Tuple, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging


class RAGPipline:
    """
    Docs
    """
    def __init__(self, db_name: str = "./chromastore.db") -> None:
        """
        """
        self.db_name = db_name
        self.documents = []

        self._embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        self._textsplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self._vectorstore = None

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents as Langchain Documents

        Args:
            documents (list of Langchain Documents): The documents to be ragged
        """
        if not all([isinstance(doc, Document) for doc in documents]):
            logging.warning("Found non-langchain document in ")
        else:
            self.documents.extend(documents)
        chunks = self._textsplitter.split_documents(self.documents)
        self._vectorstore = Chroma.from_documents(
            documents = chunks,
            embedding = self._embeddings,
            persist_directory = self.db_name
        )
        
    def load_texts(texts: List[str]) -> int:
        """
        """

