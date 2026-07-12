import os, re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.callbacks import get_openai_callback
from langchain_groq import ChatGroq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from src.api.schemas import AnswerSchema, QueryRequest, SourceSchema
from src.weaviate.query import query_all
from src.memory_agent import chat
import logging
import json
from fastapi.responses import StreamingResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
# -------------
# Retriever
# -------------
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index_name="test-index", embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k":5})

# ------------
# LLM chain
# ------------
def build_llm_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens = 4096
    )
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful RAG assistant. Answer using ONLY the retrieved context.
        Write in plain prose. Do NOT use backticks, bold, bullet points, or any markdown in your answer text.
        At the end of the end of your answer, OUTPUT a json block (and nothing after it) in this exact format:
         ```json
         {{
            "confidence":<float between 0 and 1 reflecting the confidence>,
            "citations":["[Chunk 0]","[Chunk 1]"] //replace with the actual chunk numbers
         }}
         ```
         """),
        ("human","Question:{question}\n\nContext:\n{context}\n\nProvide a structured answer."),
    ])
    return prompt | llm

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message":"Hello World!"}

@app.get("/health")
async def check_health():
    return {"status": "ok"}

@app.post("/query")
async def query(request: QueryRequest):
    question = request.question
    memory_result = chat(session_id=request.thread_id,
                         user_message=question,
                         db = request.db)
    context = memory_result["retrieved_context"]
    question = memory_result["resolved_query"]

    #1. Pinecone retrieval for sources only
    if request.db == "pinecone":
        docs = retriever.invoke(question)
    else:
    #2. Weaviate Retrieval for sources only
        raw = query_all(question,limit=5,search_type="hybrid")
        docs = [
            Document(
                page_content=obj.properties.get("text"),
                metadata = {
                    "ticker": obj.properties.get("ticker"),
                    "year": obj.properties.get("year"),
                    "quarter": obj.properties.get("quarter"),
                    "filing_type": obj.properties.get("filing_type"),
                    "sources": obj.properties.get("source"),
                }
            ) 
            for obj in raw
        ]
    

    # 2. Build context from your real chunks
    # context = "\n\n".join(
    #     f"[Chunk {i}] Source: {doc.metadata.get('source', '?')} | "
    #     f"Ticker: {doc.metadata.get('ticker', '?')} | "
    #     f"Year: {doc.metadata.get('year', '?')}\n{doc.page_content}"
    #     for i, doc in enumerate(docs)
    # )

    #3. Generate metadata here 
    sources = [
        {
            "chunk_id": i,
            "source": doc.metadata.get("source", "?"),
            "ticker": doc.metadata.get("ticker", "?"),
            "year": str(doc.metadata.get("year", "?")),
            "preview": doc.page_content[:200],
        }
    for i,doc in enumerate(docs)
    ]
    # 3. Create chain
    chain = build_llm_chain()

    #4. Streaming helper method
    async def stream_response():
        full_response = ""
        try:
            with get_openai_callback() as cb:
                async for chunk in chain.astream({"question":question,"context": context}):
                    token = chunk.content
                    full_response+=token
                    if "```json" not in full_response:
                        yield f"data: {json.dumps({"type":'token', "content": token})}\n\n"
        except Exception as e:
            yield f"data:{json.dumps({'type':'error','detail': str(e)})}\n\n"
            return

        #5. Parse confidence and citations from the JSON block
        #Default values for citations and confidence
        confidence = 0.0
        citations = []

        match = re.search(r"```json\s*(\{.*?\})\s*```",full_response,re.DOTALL)
        if match:
            try:
                meta = json.loads(match.group(1))
                confidence = meta.get("confidence", 0)
                citations = meta.get("citations", [])
            except json.JSONDecodeError as e:
                logger.info(f"Could not parse metadata: {e}")
        

        #Step 6: Build the final AnswerSchema and send it as one closing event 
        final = AnswerSchema(**{
            "answer": full_response.split("```json")[0].strip(),
            "confidence": confidence,
            "citations": citations,
            "sources": sources,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "cost_usd": cb.total_cost
        })
        
        yield f"data:{json.dumps({"type":"final", "data": final.model_dump()})}\n\n"
        yield f"data:[DONE]\n\n"
    
    #7.Return the stream to FastAPI
    return StreamingResponse(stream_response(),media_type="text/event-stream")