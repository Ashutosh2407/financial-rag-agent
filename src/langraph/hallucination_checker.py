import re, os
import guardrails as gr
from guardrails.hub import DetectPII, PromptInjectionDetector
from guardrails.types import OnFailAction
from typing import Annotated, TypedDict, List, Literal
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

class GroundingScore(BaseModel):
    grounding_score: float
    confidence_score: float

llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens = 4096,
        temperature=0
    )


def check_grounding(chunks: List[str], answer: str)-> dict:
    combined_context = "\n".join(chunks)
    judge_prompt = (
        f"Context:\n{combined_context}\n\nAnswer:\n{answer}\n\n"
        "Rate on a 0-1 scale how fully the answer is supported by the context. "
        "Return two numbers: grounding_score, confidence_score."
    )
    structured_llm = llm.with_structured_output(GroundingScore)
    results = structured_llm.invoke(judge_prompt)
    return results.grounding_score, results.confidence_score

