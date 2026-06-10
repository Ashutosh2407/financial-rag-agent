# 1)Load test.json
# 2)Generate answers for the question using My LLM
# 3)Compare the answers with test.json using RAG evaluation on metrics
#     a) faithfulness
#     b)answer_relevance
#     c)context_precision
# 4)Once the result is generated, save it in eval_dataset.json

import os, json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tracers.langchain import wait_for_all_tracers
from src.normalize_metadata import load_and_normalize
from src.ingest import load_all_docs
from src.chunking import FinanceSectionChunker,get_chunking_context
from dotenv import load_dotenv
from src.retriever import Retriever
from src.api.main import build_llm_chain
from src.api.schemas import AnswerSchema,EvalQuestionSchema
import asyncio
import logging
import time 

load_dotenv()

with open("src/eval/test_set.json", "r") as f:
    test_set = json.load(f)

os.makedirs("src/eval/datasets", exist_ok=True)

pages, manifest = load_and_normalize()
print(len(pages))
print(len(manifest))
ctx = get_chunking_context("section")
split_chunks = ctx.chunk(pages)


retriever = Retriever(docs=split_chunks, embeddings= HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2"))


MAX_CONTEXT_CHAR = 4000
async def get_answer(questions: EvalQuestionSchema, strategy:str):
    RESULTS = []
    for item in questions:
        question = item["question"]
        
        if strategy == "dense":
            #Pinecone retrieval
            active_retriever = retriever.get_dense_retriver()
        elif strategy == "sparse":
            active_retriever = retriever.get_bm25_sparse_retriver()
        elif strategy == "hybrid":
            active_retriever = retriever.get_ensemble_retiever()
        else:
            print("Please input a valid retrieval strategy.")
            break

        docs = active_retriever.invoke(question)
        
        #Build context from your real chunks
        context = "\n\n".join(
            f"[Chunk {i}] Source: {doc.metadata.get('source', '?')} | "
            f"Ticker: {doc.metadata.get('ticker', '?')} | "
            f"Year: {doc.metadata.get('year', '?')}\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )
        context = context[:MAX_CONTEXT_CHAR]
        #Generate Structured Answer
        chain = build_llm_chain()
        try:
            result= await chain.ainvoke(
                {"question":question, "context":context}
            )
        except Exception as e:
            raise ValueError(f"Could not generate answer for {question}.")
        
        RESULTS.append(
            {
                "strategy":strategy,
                "question":question,
                "answer":result.answer,
                "ground_truth": item["ground_truth"],
                "contexts": [d.page_content for d in docs],
                "category": item["category"],
                "expected_source": item["expected_source"],
                "actual_sources": result.sources,
                "target_companies": item["target_companies"],
                "reference_period": item["reference_period"],
                "citations": result.citations,
            }
        )
        #break
        #time.sleep(5)
    with open(f"src/eval/datasets/eval_dataset_{strategy}.json","w") as f:
        json.dump(RESULTS,f, indent=2)

#result_sparse = asyncio.run(get_answer(test_set["questions"],strategy="sparse"))

#result_hybrid = asyncio.run(get_answer(test_set["questions"],strategy="hybrid"))



