from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from src.chunking.base import BaseChunker
from langchain_core.documents import Document
import re

#Concrete Strategy 1
class FixedSizedChunker(BaseChunker):
    """
    Splits by a fixed character window with no structural awareness.
    Baseline — intentionally breaks tables and risk items for comparison.
    """
    def __init__(self, chunk_size=100, chunk_overlap=0):
        self._text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

    def split(self, docs):
        texts = self._text_splitter.split_documents(docs)
        return texts


#Concrete Strategy 2
class RecursiveTextChunker(BaseChunker):
    def __init__(self,chunk_size=100, chunk_overlap=0):
        self._recursivetextsplitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )

    def split(self,docs):
        texts = self._recursivetextsplitter.split_documents(docs)
        return texts

#Concrete Strategy 3
_SECTION_PATTERN = re.compile(
    r'(Item\s+\d+[A-Z]?\.\s+|'
    r'MANAGEMENT.{0,10}DISCUSSION|'
    r'RISK FACTORS|'
    r'QUANTITATIVE AND QUALITATIVE|'
    r'(?:^|\n)(?:Operator|[A-Z][a-z]+ [A-Z][a-z]+):\s)',
    re.MULTILINE | re.IGNORECASE
)

class FinanceSectionChunker(BaseChunker):
    def __init__(self,max_section_size: int = 2000, chunk_overlap: int = 200):
        self._max = max_section_size
        self._secondary = RecursiveCharacterTextSplitter(
            chunk_size = max_section_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split(self,docs):
        result = []
        for doc in docs:
            raw_sections = _SECTION_PATTERN.split(doc.page_content)
            sections = [s.strip() for s in raw_sections if s and len(s.strip()) >= 50]
            for section in sections:
                if len(section) <=2000:
                    result.append(Document(page_content=section, metadata = doc.metadata))
                else:
                    result.extend(self._secondary.create_documents([section],metadatas=[doc.metadata]))
        
        return result

