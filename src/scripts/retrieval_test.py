import os,json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tracers.langchain import wait_for_all_tracers
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.Logger(__name__)

QUESTIONS = [
    "What were the key risk factors for JPMorgan in 2025?",
    "What did JPMorgan management say about net interest income guidance?",
    "How did JPMorgan's CET1 capital ratio change in 2025?",
    "What were the main drivers of JPMorgan's revenue growth in 2025?",
    "What credit loss provisions did JPMorgan disclose in 2025?",
]

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(embedding=embeddings,index_name="test-index")
retriever = vectorstore.as_retriever(search_kwargs={"k":5})
results = {}
for q in QUESTIONS:
    docs = retriever.invoke(q)
    results[q] = [
        {
            "chunk_id":i,
            "content_preview":doc.page_content[:300],
            "metadat": doc.metadata,
            "char_length": len(doc.page_content)
        }
    for i,doc in enumerate(docs)]
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    for i,doc in enumerate(docs):
        print(f"\n [Chunk {i+1}] Source: {doc.metadata.get("source","?")} | "
              f"Ticker: {doc.metadata.get("ticker","?")} | "
              f"Year: {doc.metadata.get("year","?")})")
        print(f"Preview: {doc.page_content[:200]}...")

#save inspection report
with open("data/retrieval_inspection_report_day4.json", "w") as f:
    json.dump(results,f,indent=2)

logging.info("\n✅ Saved to data/retrieval_inspection_day4.json")
wait_for_all_tracers()