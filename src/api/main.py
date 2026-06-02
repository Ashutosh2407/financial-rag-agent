import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from schemas import AnswerSchema

load_dotenv()


app = FastAPI()
# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index="test-index", embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k":5})

# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------
class RequestBody(BaseModel):
    question:str = Field(..., min_length=3)
    top_k:int = Field(default=5,ge=1,le=20)


@app.get("/")
async def root():
    return {"message":"Hello World!"}

@app.get("/health")
async def check_health():
    return {"status": "ok"}