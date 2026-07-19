import asyncio,os,re
from langgraph.graph import StateGraph,START,END
from langgraph.types import Command, interrupt
from typing import Annotated, TypedDict, List, Literal, Dict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from src.retriever import retriever
from src.langraph.guardrails_node import input_guardrail,output_guardrail
from src.langraph.hallucination_checker import check_grounding
from dotenv import load_dotenv

load_dotenv()

CONFIDENCE_THRESHOLD = 0.75

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
        temperature=0,
        streaming=False
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
    chunk_sources: List[Dict]
    web_sources: List[Dict]
    grade:float
    confidence: float
    blocked: bool
    blocked_reason: str
    approved: bool
    gen_count: int = 0

def retrieve(state: AnswerState)->dict:
    query = state["current_query"]
    docs = retriever.invoke(query)
    chunks = [d.page_content for d in docs]
    chunk_sources = [
        {
            "type": "chunk",
            "chunk_id": i,
            "source": d.metadata.get("source", "?"),
            "ticker": d.metadata.get("ticker", "?"),
            "year": str(d.metadata.get("year", "?")),
            "preview": d.page_content[:200],
        }
        for i, d in enumerate(docs)
    ]
    return {"chunks":chunks,"chunk_sources":chunk_sources}

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

async def guardrail_wrapper(state: AnswerState, mode: Literal["input","output"])-> dict:
    if mode == "input":
        safe,reason = input_guardrail(state["current_query"])
        if not safe:
            return {"blocked":True, "blocked_reason": reason}
        return {"blocked": False}
    elif mode == "output":
        text = "".join(state["answer"])
        safe,reason = output_guardrail(answer = text)
        if not safe:
            return {"blocked":True, "blocked_reason": reason}      
        return {"blocked": False}
    raise ValueError(f"Unknown guardrails mode: {mode}")

def tavily(state: AnswerState)->dict:
    print("Tavily search begins...")
    query = state["current_query"]
    tool = TavilySearchResults(
        api_key = os.environ.get("TAVILY_API_KEY"),
        max_results=5,
        include_answer=True,
        include_raw_content=True,
    )
    results = tool.invoke({"query": query})
    web_context = "\n\n".join(f"[Web Result {i}] {r['content']}" for i,r in enumerate(results))
    web_sources = [
        {
            "type": "web",
            "result_id": i,
            "title": r.get("title", "Untitled"),
            "url": r.get("url", ""),
            "preview": r.get("content", "")[:300]
        }
    for i, r in enumerate(results)]
    return {"chunks": [web_context],"web_sources":web_sources}

async def generator(state: AnswerState)->dict:
    chain = build_llm_chain()
    result = []
    async for chunk in chain.astream({"question":state["current_query"],"context": state["retrieved_context"]}):
        result.append(chunk.content)
    return {"answer": "".join(result),"gen_count": state.get("gen_count",0)+1}

def hallucination_check(state: AnswerState)->dict:
    """Verify answer is grounded in retrieved chunks using an LLM-as-judge check."""
    grounding_score,confidence_score=check_grounding(state["chunks"],state["answer"])
    return {"confidence":confidence_score}

def route_after_hallucination_check(state: AnswerState)->str:
    if state["confidence"] < CONFIDENCE_THRESHOLD and state.get("gen_count",0) <2:
        return "generator_node"
    elif state["confidence"] < CONFIDENCE_THRESHOLD:
        return "human_review_node"
    return END

def human_review(state:AnswerState)-> Command:
    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        return Command(
            update = AnswerState(answer=state["answer"],approved= True),
            goto="guardrails_output"
        )
    else:
        print("Interrupted.")
        decision = interrupt({
            "reason": "low_confidence",
            "confidence": state["confidence"],
            "query": state["current_query"],
            "draft_answer": state["answer"],
            "supporting_chunks": state["chunks"],
        })
        if decision == "approved":
            answer = state["answer"]
            approved = True
            next_node =  "guardrails_output"
        else:
            answer = "Answer could not be found. Analyst rejects."
            approved = False
            next_node = END
        return Command(
            update=AnswerState(answer=answer,approved= approved),
            goto=next_node
        )

async def guardrails_input_node(state: AnswerState)-> dict:
    return await guardrail_wrapper(state,"input")

async def guardrails_output_node(state:AnswerState)-> dict:
    return await guardrail_wrapper(state,"output")


def build_graph():
    builder = StateGraph(state_schema = AnswerState)
    
    builder.add_node("guardrails_input", guardrails_input_node)
    builder.add_node("guardrails_output", guardrails_output_node)
    builder.add_node("retriever_node", retrieve)
    builder.add_node("grader_node",grader)
    builder.add_node("contextualize_node",contextualize)
    builder.add_node("generator_node", generator)
    builder.add_node("tavily_node", tavily)
    builder.add_node("hallucination_checker_node", hallucination_check)
    builder.add_node("human_review_node", human_review)
    
    builder.add_conditional_edges("guardrails_input",
    lambda s: "blocked" if s.get("blocked") else "continue",
    {"blocked": END, "continue":"retriever_node"}
    )
    
    builder.add_conditional_edges("guardrails_output",
    lambda s: "blocked" if s.get("blocked") else "ok",
    {"blocked": END, "ok":END}
    )

    builder.add_edge(START,"guardrails_input")
    builder.add_edge("retriever_node", "grader_node")
    builder.add_conditional_edges("grader_node", conditional_edge)
    builder.add_edge("tavily_node", "contextualize_node")
    builder.add_edge("contextualize_node","generator_node")
    builder.add_edge("generator_node","hallucination_checker_node")
    builder.add_conditional_edges("hallucination_checker_node",route_after_hallucination_check)
    builder.add_edge("guardrails_output", END)
    return builder.compile(checkpointer=MemorySaver())

graph = build_graph()
if __name__ == "__main__":
    config = {
        "configurable":{
            "thread_id": "test_run_1"
        }
    }
    result = asyncio.run(
        graph.ainvoke({"current_query": "When is apple i phone product launch?", 
                    "retrieved_context": "", 
                    "answer": [""]},
                    config=config)
    )
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value  #dict passed to interrupt
        print("Needs review:", payload)
        verdict = input("approve? ")
        final = asyncio.run(
            graph.ainvoke(
                Command(resume=verdict),
                config=config
            )
        )
        print("Final answer....\n")
        print(final["answer"])