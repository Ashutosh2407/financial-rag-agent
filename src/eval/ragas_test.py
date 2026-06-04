# 1)Load test.json
# 2)Generate answers for the question using My LLM
# 3)Compare the answers with test.json using RAG evaluation on metrics
#     a) faithfulness
#     b)answer_relevance
#     c)context_precision
# 4)Once the result is generated, save it in results.csv

import os, json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tracers.langchain import wait_for_all_tracers
from dotenv import load_dotenv
from src.api.main import build_llm_chain
from src.api.schemas import AnswerSchema,EvalQuestionSchema
import asyncio
import logging

load_dotenv()

with open("src/eval/test_set.json", "r") as f:
    test_set = json.load(f)

os.makedirs("src/eval/datasets", exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(embedding=embeddings,index_name="test-index")
retriever = vectorstore.as_retriever(search_kwargs={"k":5})

async def get_answer(request: EvalQuestionSchema)->AnswerSchema:
    results = []
    for item in request:
        question = item["question"]
        #Pinecone retrieval
        docs = retriever.invoke(question)
    
        #Build context from your real chunks
        context = "\n\n".join(
            f"[Chunk {i}] Source: {doc.metadata.get('source', '?')} | "
            f"Ticker: {doc.metadata.get('ticker', '?')} | "
            f"Year: {doc.metadata.get('year', '?')}\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )
        #Generate Structured Answer
        chain = build_llm_chain()
        try:
            result: AnswerSchema = await chain.ainvoke(
                {"question":question, "context":context}
            )
        except Exception as e:
            raise ValueError("Not conforming to answerschema.")
        
        results.append(result.model_dump())
        
    with open("src/eval/datasets/eval_dataset.json","w") as f:
        json.dump(results,f, indent=2)

result = asyncio.run(get_answer(test_set["questions"]))

