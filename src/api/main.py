import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from src.api.schemas import AnswerSchema, QueryRequest

load_dotenv()


app = FastAPI()
# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index_name="test-index", embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k":5})

# ---------------------------------------------------------------------------
# LLM chain
# ---------------------------------------------------------------------------
def build_llm_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        api_key=os.environ.get("GROQ_API_KEY")
    )
    structured_llm = llm.with_structured_output(AnswerSchema)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful RAG assistant. Answer using ONLY the retrieved context."),
        ("human","Question:{question}\n\nContext:\n{context}\n\nProvide a structured answer."),
    ])
    return prompt | structured_llm

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message":"Hello World!"}

@app.get("/health")
async def check_health():
    return {"status": "ok"}

@app.post("/query", response_model=AnswerSchema)
async def query(request: QueryRequest)-> AnswerSchema:
    question = request.question
    # 1. Real Pinecone retrieval
    docs = retriever.invoke(question)
    # 2. Build context from your real chunks
    context = "\n\n".join(
        f"[Chunk {i}] Source: {doc.metadata.get('source', '?')} | "
        f"Ticker: {doc.metadata.get('ticker', '?')} | "
        f"Year: {doc.metadata.get('year', '?')}\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    # 3. Generate structured answer
    chain = build_llm_chain()
    try:
        result: AnswerSchema = await chain.ainvoke(
            {"question":question, "context":context}
        )
    except Exception as e:
        raise HTTPException(status_code=502,detail=f"LLM call failed: {e}")

    # 4. Attach Real source metadata
    result.sources.extend([
        {
            "chunk_id": i,
            "source": doc.metadata.get("source", "?"),
            "ticker": doc.metadata.get("ticker", "?"),
            "year": doc.metadata.get("year", "?"),
            "preview": doc.page_content[:200],
        }
    for i,doc in enumerate(docs)
    ])
    return result

    