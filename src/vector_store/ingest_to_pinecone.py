#Chunk, embed and upsert
import os
from langchain_community.document_loaders import PyPDFLoader
from src.chunking import FinanceSectionChunker, get_chunking_context
from src.ingest import load_all_docs
from src.vector_store.pinecone_client import get_index
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.normalize_metadata import load_and_normalize
import src.corpus_config

path = "/Users/ashutoshwagh/Desktop/projects/financial-rag-agent/data/goldman_sachs.pdf"

#docs = load_all_docs(path)
# loader = PyPDFLoader(path)
# docs = loader.load()
index = get_index(name = "test-index")
index.delete(delete_all=True)
print("🗑️  Index cleared")

pages, manifest = load_and_normalize()
print(len(pages))
print(len(manifest))
ctx = get_chunking_context("section")
split_chunks = ctx.chunk(pages)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = get_index(name = "test-index")

vector_store = PineconeVectorStore.from_documents(documents=split_chunks, embedding=embeddings,index_name="test-index")

print(f"Uploaded {len(pages)} pages.")
print(f"Uploaded {len(split_chunks)} chunks.")