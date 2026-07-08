from langgraph.graph import StateGraph,START,END
from typing import Annotated, TypedDict, List
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from pydantic import BaseModel, Field
from src.retriever import retriever
import asyncio
import os,re
from dotenv import load_dotenv

load_dotenv()

def build_llm_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens = 4096,
        temperature=0
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


class GradeDocuments(BaseModel):
    score: float = Field(description="Relevance score from 0 to 1",ge=0,le=1)

def grader_llm_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens = 4096,
        temperature=0
    )
    structured_llm = llm.with_structured_output(GradeDocuments)
    prompt = ChatPromptTemplate.from_messages([(
        "system","""You are a grader assessing relevance of the retrieved documents to a user question. Give 
        a relevance score between 0 and 1 where 1 means highly relevant."""),
        ("human","Question: {question}\n\nDocument:\n{document}\n\nRate relevance.")])
    return prompt | structured_llm


class AnswerState(TypedDict):
    answer: str
    retrieved_context: str 
    current_query:str
    chunks: List[str]
    grade:float


def retrieve(state: AnswerState)->dict:
    query = state["current_query"]
    docs = retriever.invoke(query)
    chunks = [d.page_content for d in docs]
    return {"chunks":chunks}

def contextualize(state: AnswerState)-> dict:
    chunks = state["chunks"]
    if isinstance(chunks, list):
        context = "\n---\n".join(chunks)
    else:
        context = chunks
    return {"retrieved_context": context}
    
def grader(state: AnswerState)-> dict:
    grader_chain = grader_llm_chain()
    docs = state["chunks"]
    scores = []
    question = state["current_query"]
    for document in docs:
        result = grader_chain.invoke({"question": question, "document": document})
        scores.append(result.score)
    avg_score = sum(scores)/len(scores) if scores else 0.0
    return {"grade":avg_score}

def conditional_edge(state: AnswerState)-> str:
    score = state["grade"]
    if score >=0.6:
        print(f"[ROUTING] grade={score:.2f} → contextualize (using retrieved chunks)")
        return "contextualize_node"
    else:
        print(f"[ROUTING] grade={score:.2f} → tavily (web search fallback)")
        return "tavily_node"



def tavily(state: AnswerState)->dict:
    query = state["current_query"]
    tool = TavilySearchResults(
        api_key = os.environ.get("TAVILY_API_KEY"),
        max_results=5,
        include_answer=True,
        include_raw_content=True,
    )
    results = tool.invoke({"query": query})
    web_context = "\n\n".join(f"[Web Result {i}] {r['content']}" for i,r in enumerate(results))
    return {"chunks": [web_context]}

async def generator(state: AnswerState)->dict:
    chain = build_llm_chain()
    result = []
    async for chunk in chain.astream({"question":state["current_query"],"context": state["retrieved_context"]}):
        result.append(chunk.content)
    return {"answer": result}

def build_graph():
    builder = StateGraph(state_schema = AnswerState)
    
    builder.add_node("retriever_node", retrieve)
    builder.add_node("grader_node",grader)
    builder.add_node("contextualize_node",contextualize)
    builder.add_node("generator_node", generator)
    builder.add_node("tavily_node", tavily)

    builder.add_edge(START,"retriever_node")
    builder.add_edge("retriever_node", "grader_node")
    builder.add_conditional_edges("grader_node", conditional_edge)
    builder.add_edge("tavily_node", "contextualize_node")
    builder.add_edge("contextualize_node","generator_node")
    builder.add_edge("generator_node", END)

    return builder.compile()

graph = build_graph()
result = asyncio.run(
    graph.ainvoke({"current_query": "What is the most sold product of apple in 2026?", "retrieved_context": "", "answer": [""]})
)
print(result)