# 1)Load test.json
# 2)Generate answers for the question using My LLM
# 3)Compare the answers with test.json using RAG evaluation on metrics
#     a)faithfulness
#     b)answer_relevance
#     c)context_precision
# 4)Once the result is generated, save it in eval_dataset.json

import os, json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.normalize_metadata import load_and_normalize
from src.corpus_config import CORPUS_10K
from src.ingest import load_all_docs
from src.chunking import FinanceSectionChunker,get_chunking_context
from src.weaviate.query import query_all
from src.retriever import Retriever
from src.api.schemas import AnswerSchema,EvalQuestionSchema
from dotenv import load_dotenv
import asyncio
import logging
import time 

load_dotenv()

MAX_CONTEXT_CHAR = 4000

with open("src/eval/test_set.json", "r") as f:
    test_set = json.load(f)

os.makedirs("src/eval/datasets", exist_ok=True)



def build_llm_chain_to_generate_answer():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens = 4096
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", """You are a helpful RAG assistant. Answer using ONLY the retrieved context."""),
        ("human","Question:{question}\n\nContext:\n{context}\n\nProvide a structured answer.")]
    )
    structured_llm = llm.with_structured_output(AnswerSchema)
    return prompt | structured_llm


async def get_answer(questions: EvalQuestionSchema, strategy:str):
    #Set up Retriever object
    pages, manifest = load_and_normalize(corpus= CORPUS_10K)
    print(len(pages))
    print(len(manifest))
    ctx = get_chunking_context("section")
    split_chunks = ctx.chunk(pages)
    retriever = Retriever(docs=split_chunks, embeddings= HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2"))
    
    for item in questions:
        question = item["question"]
        
        if strategy == "dense":
            #Pinecone retrieval
            active_retriever = retriever.get_dense_retriever()
        elif strategy == "sparse":
            active_retriever = retriever.get_bm25_sparse_retriever()
        elif strategy == "hybrid":
            active_retriever = retriever.get_ensemble_retiever()
        elif strategy == "compression":
            active_retriever = retriever.get_contextual_compression_retriever()
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
        chain = build_llm_chain_to_generate_answer()
        try:
            result= await chain.ainvoke(
                {"question":question, "context":context}
            )
            print(result)
        except Exception as e:
            raise ValueError(f"Could not generate answer for {question}.")
        
        row = {
                "strategy":strategy,
                "question":question,
                "answer":result.answer,
                "ground_truth": item["ground_truth"],
                "contexts": [d.page_content for d in docs],
                "category": item["category"],
                "expected_source": item["expected_source"],
                "actual_sources": [s.model_dump() for s in result.sources],
                "target_companies": item["target_companies"],
                "reference_period": item["reference_period"],
                "citations": result.citations,
            }
        
        with open(f"src/eval/datasets/eval_dataset_{strategy}.json", "a") as f:
            f.write(json.dumps(row) + "\n")
        time.sleep(5)


async def get_answer_weaviate(questions: EvalQuestionSchema,strategy:str,search_type:str="dense"):
    i=1
    for item in questions:
        question = item["question"]
        chunk_results = query_all(query =question,limit=2,search_type=search_type)
        #Build context from your real chunks
        context = "\n\n".join(
            f"[Chunk {i}] Source: {r.properties.get('source', '?')} | "
            f"Ticker: {r.properties.get('ticker', '?')} | "
            f"Year: {r.properties.get('year', '?')}\n{r.properties.get('text','Content Not Available.')}"
            for i, r in enumerate(chunk_results)
        )
        context = context[:MAX_CONTEXT_CHAR]
        #Generate Structured Answer
        chain = build_llm_chain_to_generate_answer()
        try:
            llm_result = await chain.ainvoke({
                "question":question,
                "context": {context}
            })
        except Exception as e:
            raise ValueError(f"Could not generate answer for {question}.")
        
        row = {
                "strategy":"weaviate_"+search_type,
                "question":question,
                "answer":llm_result.answer,
                "ground_truth": item["ground_truth"],
                "contexts": [r.properties.get("text") for r in chunk_results],
                "category": item["category"],
                "expected_source": item["expected_source"],
                "actual_sources": [s.properties.get("source") for s in chunk_results],
                "target_companies": item["target_companies"],
                "reference_period": item["reference_period"],
                "citations": llm_result.citations,
            }
        with open(f"src/eval/datasets/eval_dataset_{strategy}_{search_type}.json", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"Question {i} is done.")
        i+=1
        time.sleep(5)

    





#result_sparse = asyncio.run(get_answer(test_set["questions"],strategy="sparse"))

#result_hybrid = asyncio.run(get_answer(test_set["questions"],strategy="hybrid"))

#result_compression = asyncio.run(get_answer(test_set["questions"],strategy="compression"))

#result_weaviate = asyncio.run(get_answer_weaviate(test_set["questions"],strategy="weaviate",search_type="dense"))

#result_weaviate = asyncio.run(get_answer_weaviate(test_set["questions"],strategy="weaviate",search_type="hybrid"))



