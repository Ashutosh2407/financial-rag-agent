# financial-rag-agent-aws

## Business Problem
Financial analysts and compliance teams spend hours manually reviewing 10-Ks,
earnings transcripts, and regulatory filings to answer ad-hoc questions.
Answers are inconsistent, slow, and hard to audit. This assistant provides
grounded, cited answers from a structured corpus of SEC filings.

## Corpus Description
- **Sources:** SEC EDGAR 10-K filings.
- **Companies:** 5 financial institutions (Apple, JPMorgan, Goldman Sachs, Microsoft, Tesla).
- **Filing years:** 2025.
- **Ingestion:** PyPDFLoader → RecursiveCharacterTextSplitter.
- **Metadata per chunk:** ticker, company, filing_type, year, section, source_url.
- **Vector store:** Pinecone (sentence-transformers/all-MiniLM-L6-v2, 384 dims).
- **Total chunks indexed:** ~2500 chunks across 5 documents.

## Day 4 Retrieval Test Results
![Langchain Dashboard](langchaindashboard.png)

## Retrieval Strategy Benchmark
| Strategy              | Faithfulness | Answer Relevance | Context Precision |
| -------------------   | ------------ | ---------------- | ----------------- |
| Dense (Pinecone)      |  0.44        | 0.62             | 0.48              |
| Sparse (BM25)         |  0.70        | 0.77             | 0.44              | 
| Hybrid (BM25 + Dense) |  0.78        | 0.81             | 0.45              |
| Compresession (rerank)|  0.68        | 0.87             | 0.62              |
| Weaviate	            |  0.67	       | 0.72	            | 0.39              |

## Demo 1
![Demo](https://github.com/user-attachments/assets/e84c8cd8-b5ae-4bb0-a876-7037f83bc7e9)

## Demo 2 : Multi-turn chat
![Demo](https://github.com/user-attachments/assets/b9049608-6765-4678-a5e5-a405cfb4db9a)
