#Resolves follow up queries.
import re
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from src.weaviate.query import query_all
from src.retriever import retriever


class ConversationState(TypedDict):
    messages: Annotated[list,add_messages] #full chat history, append only
    retrieved_context: str #latest chunks from weaviate
    current_query:str   #rewritten query for follow ups
    db:str #Database: Weaviate or Pinecone

def retrieve_context(state:ConversationState)-> dict:
    """
    Takes the contextualized query from contextualize_query and 
    retrieves appropriate document chunks from weaviate/pinecone.
    """
    query = state["current_query"]
    db = state.get("db", "weaviate")
    if db == "weaviate":
        results = query_all(query=query)
        chunks = [r.properties.get("text") for r in results]
    else:
        docs = retriever.invoke(query)
        chunks = [d.page_content for d in docs]
    context = "\n---\n".join(chunks)
    return {"retrieved_context": context}


FOLLOWUP_RE = re.compile(
    r"\b(they|those|these|it|that|them|compare|same|similar|how about|what about)\b",
    re.IGNORECASE
)

def contextualize_query(state:ConversationState)-> dict:
    """
    Adds previous messages/queries to the latest query for 
    context if it is a follow up question.
    """
    messages = state["messages"]
    latest_message = messages[-1].content.strip()
    #Find the last AI reply
    #prior_ai = [m.content for m in messages[:-1] if isinstance(m,AIMessage)]
    prior_ai = messages[:-1]

    is_followup = bool(FOLLOWUP_RE.search(latest_message)) and bool(prior_ai)

    if is_followup:
        #Append a short snippet of last answer for context
        context_hint = prior_ai[-1].content[:200]
        resolved = f"{latest_message}  [prior:  {context_hint}]"
    else:
        resolved = latest_message

    return {"current_query": resolved}


def build_graph():
    """
    START -> CONTEXTUALIZE -> RETRIEVE -> END
    """
    builder = StateGraph(state_schema=ConversationState)

    builder.add_node("contextualize", contextualize_query)
    builder.add_node("retrieve", retrieve_context)

    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "retrieve")
    builder.add_edge("retrieve", END)

    return builder.compile(checkpointer=MemorySaver())

graph = build_graph()

def chat(session_id:str, user_message:str, db:str = "weaviate") -> dict:
    config = {
        "configurable":{
            "thread_id": session_id
        }
    }

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "retrieved_context":"",
            "current_query": user_message,
            "db":db
        },
        config = config
    )

    return {
        "query": user_message,
        "resolved_query": result["current_query"],
        "retrieved_context": result["retrieved_context"]
    }
