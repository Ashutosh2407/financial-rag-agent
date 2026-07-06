from langgraph.graph import StateGraph,START,END
from typing import Annotated, TypedDict
from src.api.schemas import AnswerSchema
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.retriever import retriever
import asyncio
import os,re

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

class AnswerState(TypedDict):
    answer: str
    retrieved_context: str 
    current_query:str

def retriever_node(state: AnswerState)-> dict:
    query = state["current_query"]
    docs = retriever.invoke(query)
    chunks = [d.page_content for d in docs]
    context = "\n---\n".join(chunks)
    return {"retrieved_context": context}

async def generator(state: AnswerState)->dict:
    chain = build_llm_chain()
    result = []
    async for chunk in chain.astream({"question":state["current_query"],"context": state["retrieved_context"]}):
        result.append(chunk.content)
    return {"answer": result}

def build_graph():
    builder = StateGraph(state_schema = AnswerState)
    
    builder.add_node("retrieve",retriever_node)
    builder.add_node("generate", generator)

    builder.add_edge(START,"retrieve")
    builder.add_edge("retrieve","generate")
    builder.add_edge("generate", END)

    return builder.compile()

graph = build_graph()
result = asyncio.run(
    graph.ainvoke({"current_query": "What are the company's main risks?", "retrieved_context": "", "answer": [""]})
)
print(result)