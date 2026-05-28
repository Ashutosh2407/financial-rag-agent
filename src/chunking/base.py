from abc import ABC, abstractmethod
from langchain_core.documents import Document
import statistics

class Basechunker(ABC):
    """
    Strategy Interface.
    All chunkers must implement split().
    Context programs to this interface only — never to concrete classes.
    """
    @abstractmethod
    def split(self, docs: list[Document]) -> list[Document]:
        pass

    def stats(self, docs: list[Document])-> dict:
        """Summary dict — used for MLflow logging in Week 10."""
        chunks = self.split(docs)
        lengths = [len(c.page_content) for c in chunks]
        return {
            "strategy": self.__class__.__name__,
            "chunk_count": len(chunks),
            "avg_len": statistics.mean(lengths),
            "min_len": min(lengths) if lengths else 0,
            "max_len": max(lengths) if lengths else 0,
        }
