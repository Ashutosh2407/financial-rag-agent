import weaviate
from tqdm import tqdm
from weaviate.collections.collection import Collection
from weaviate.classes.config import Configure,Property,DataType
from langchain_huggingface import HuggingFaceEmbeddings
from src.normalize_metadata import load_and_normalize
from src.chunking import get_chunking_context
from src.corpus_config import CORPUS_10K,CORPUS_INTERNAL_RESEARCH,CORPUS_EARNINGS_TRANSCRIPT
import os
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

def create_schema(client):
    #10-K
    if not client.collections.exists("SECFiling"):
            client.collections.create(
                name = "SECFiling",
                vectorizer_config=Configure.Vectorizer.none(),
                properties = [
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="ticker", data_type=DataType.TEXT),
                    Property(name="company", data_type=DataType.TEXT),
                    Property(name="year", data_type=DataType.INT),
                    Property(name="filing_type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ]
            )
            print("✅ SECFiling schema created.")
    #Earnings Transcript
    if not client.collections.exists("EarningsTranscript"):
        client.collections.create(
            name = "EarningsTranscript",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="ticker", data_type=DataType.TEXT),
                Property(name="company", data_type=DataType.TEXT),
                Property(name="year", data_type=DataType.INT),
                Property(name="quarter", data_type=DataType.TEXT),
                Property(name="filing_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )
        print("✅ EarningsTranscript schema created.")
    #Internal Research/Policy Document
    if not client.collections.exists("PolicyDocument"):
        client.collections.create(
            name = "PolicyDocument",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="ticker", data_type=DataType.TEXT),
                Property(name="company", data_type=DataType.TEXT),
                Property(name="year", data_type=DataType.INT),
                Property(name="quarter", data_type=DataType.TEXT),
                Property(name="filing_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )
        print("✅ PolicyDocument Schema created.")

def load_docs(corpus):
    #load pages and add uniform metadata to them.
    pages,manifest = load_and_normalize(corpus=corpus)
    return pages

def chunk_docs(pages,strategy):
    ctx = get_chunking_context(strategy = strategy)
    split_chunks = ctx.chunk(docs = pages)
    return split_chunks

def ingest_to_weaviate(db: Collection,chunks, doc_type):   
    with db.batch.fixed_size(batch_size = 200) as batch:
        for i, chunk in enumerate(tqdm(chunks)):
            if doc_type == "sec_filing":
                doc_object = {
                    "text": chunk.page_content,
                    **chunk.metadata
                }
            elif doc_type == "internal_policy":
                doc_object = {
                    "text": chunk.page_content,
                    "company": chunk.metadata.get("company"),
                    "ticker": chunk.metadata.get("ticker"),
                    "filing_type": chunk.metadata.get("filing_type"),
                    "quarter": chunk.metadata.get("quarter"),
                    "year": chunk.metadata.get("year"),
                    "chunk_index": i
                }
            elif doc_type == "earnings_call_transcript":
                doc_object = {
                    "text": chunk.page_content,
                    "company": chunk.metadata.get("company"),
                    "ticker": chunk.metadata.get("ticker"),
                    "filing_type": chunk.metadata.get("filing_type"),
                    "quarter": chunk.metadata.get("quarter"),
                    "year": chunk.metadata.get("year"),
                    "chunk_index": i
                }

            batch.add_object(
                  properties=doc_object,
                  vector=embedding.embed_query(doc_object["text"]),
            )

if __name__ == "__main__":
    try:
        client = weaviate.connect_to_local()
        create_schema(client=client)
    except Exception as e:
        print(f"Exception: {e}")
        exit(1)
    corpuses = [CORPUS_10K,CORPUS_INTERNAL_RESEARCH,CORPUS_EARNINGS_TRANSCRIPT]
    for corpus in corpuses:
        pages, manifest = load_and_normalize(corpus)
        chunks = chunk_docs(pages= pages, strategy="section")
        if corpus == CORPUS_INTERNAL_RESEARCH:
            collection = client.collections.get("PolicyDocument")
            ingest_to_weaviate(collection,chunks=chunks,doc_type="internal_policy")
        elif corpus == CORPUS_EARNINGS_TRANSCRIPT:
            collection = client.collections.get("EarningsTranscript")
            ingest_to_weaviate(collection,chunks=chunks,doc_type="earnings_call_transcript")
        elif corpus == CORPUS_10K:
            collection = client.collections.get("SECFiling")
            ingest_to_weaviate(collection,chunks=chunks,doc_type="sec_filing")
        
    client.close()




