"""
Guardrails node for financial-rag-agent-aws
Enforces:
  (1) INPUT guard — prompt injection + out-of-scope detection
  (2) OUTPUT guard — financial disclaimer injection + PII blocking
"""
from typing import TypedDict, Literal
import re
import guardrails as gr
from guardrails.hub import DetectPII, PromptInjectionDetector
from guardrails.types import OnFailAction

# ---------------------------------------------------------------------
# 1. INPUT GUARD: prompt injection + scope check
# ---------------------------------------------------------------------

FINANCIAL_SCOPE_KEYWORDS = [
    "stock", "equity", "bond", "portfolio", "invest", "market", "fund",
    "dividend", "etf", "rate", "yield", "risk", "asset", "valuation",
    "earnings", "balance sheet", "income statement", "cash flow",
    "tax", "retirement", "401k", "ira", "loan", "mortgage", "credit",
    "profit","profits","sell","sales","product","revenue", "sales",
    "company","earnings report","quarterly results","product sales",
]

input_guard = gr.Guard().use(
    PromptInjectionDetector(on_fail=OnFailAction.EXCEPTION)
)

def _is_out_of_scope(query:str)-> bool:
    q = query.lower()
    return not any(kw in q for kw in FINANCIAL_SCOPE_KEYWORDS)

def input_guardrail(query:str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason_if_blocked)
    """
    #Check prompt injection.
    try:
        input_guard.validate(query)
    except Exception as e:
        return False, f"Prompt injection detected: {e}"
    
    if _is_out_of_scope(query):
        return (False,
                "Query appears out of scope for this financial assistant. "
                "Please ask a question related to investing, markets."
                )
    return (True,"")

# ---------------------------------------------------------------------
# 2. OUTPUT GUARD: PII block + mandatory financial disclaimer
# ---------------------------------------------------------------------
output_guard = gr.Guard().use(
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail=OnFailAction.EXCEPTION)
)

DISCLAIMER = (
    "\n\n---\n*This response is for informational purposes only and does "
    "not constitute financial, investment, or tax advice. Consult a "
    "licensed financial advisor before making decisions.*"
)

def output_guardrail(answer:str)-> tuple[bool,str]:
    """
    Validates PII, then appends disclaimer.
    Returns (is_safe, final_answer_or_block_reason)
    """
    #Check PII.
    try:
        output_guard.validate(answer)
    except Exception as e:
        return (False, f"Response blocked. PII detected. {e}")
    
    if DISCLAIMER.strip() not in answer:
        answer = answer.rstrip() + DISCLAIMER

    return (True, answer)