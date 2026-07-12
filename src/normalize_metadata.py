import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from src.corpus_config import CORPUS_10K,CORPUS_INTERNAL_RESEARCH,CORPUS_EARNINGS_TRANSCRIPT
from src.ingest import DIR_PATH
import logging
import time

logger = logging.getLogger(__name__)

def build_chunk_id(meta:dict,page:str) -> str:
    quarter = f"_{meta["quarter"]}" if meta["quarter"] else ""
    return f"{meta["ticker"]}_{meta["filing_type"].replace('-','').upper()}_{meta['year']}{quarter}_p{page}"

def load_and_normalize(corpus)->list[dict]:
    all_pages = []
    manifest_entries = []
    
    for doc_meta in corpus:
        loader = PyPDFLoader(doc_meta["local_path"])
        pages = loader.load()

        for doc in pages:
            page_num = doc.metadata.get("page",0)
            normalized = {
                "ticker":doc_meta["ticker"],
                "company": doc_meta["company"],
                "filing_type": doc_meta["filing_type"],
                "quarter": doc_meta["quarter"],
                "year": doc_meta["year"],
                "source_url": doc_meta["source_url"],
                "page": page_num,
                "chunk_id": build_chunk_id(doc_meta,page_num)
            }
            doc.metadata.update(normalized)
            all_pages.append(doc)

        # Manifest: one entry per source document
        manifest_entries.append({
            **{k:v for k,v in doc_meta.items() if k != "local_path"},
            "total_pages": len(pages),
            "chunks_loaded":len(pages)
        })

    return all_pages,manifest_entries

def save_manifest(entries:list[dict], path="src/manifest_earnings_transcript.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f:
        json.dump(entries,f,indent = 2)
    logger.info(f"✅ Manifest saved → {path} ({len(entries)} documents)")
    
def inspect_10k_chunks(chunks, ticker="JPM", n=5):
    filing_chunks = [c for c in chunks
                     if c.metadata["ticker"] == ticker
                     and c.metadata["filing_type"] == "10-K"]


    print(f"\n=== {ticker} 10-K — {len(filing_chunks)} pages ===\n")
    for chunk in filing_chunks[:n]:
        text = chunk.page_content
        print(f"--- Page {chunk.metadata['page']} | chunk_id: {chunk.metadata['chunk_id']} ---")
        print(text[:400])

        # Flag structural anomalies
        if any(x in text for x in ["|", "----", "$", "%"]):
            print("⚠️  TABLE CONTENT DETECTED — likely split mid-row")
        if "risk" in text.lower() or "item 1a" in text.lower():
            print("📌 RISK SECTION — may span 10–20 pages; consider larger chunk window")
        print()

if __name__ == "__main__":
    start = time.time()
    pages, manifest = load_and_normalize(CORPUS_EARNINGS_TRANSCRIPT)
    save_manifest(manifest)
    print(f"Total pages loaded: {len(pages)}")
    end = time.time()
    # Inspect first page metadata
    print("\n--- Sample page metadata ---")
    print(json.dumps(pages[0].metadata, indent=2))
    logger.info(f"Time used: {end-start}")
    #logger.info("INSPECTING 10-K PAGES")
    #inspect_10k_chunks(pages, ticker="JPM", n=5)

    
