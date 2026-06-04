from pydantic import BaseModel,Field, decorator, field_validator
from typing import Literal

class AnswerSchema(BaseModel):
    answer:str
    confidenece: float = Field(ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory = list)
    sources: list[str] = Field(default_factory= list)
    cost_usd: float = Field(ge=0.0)

class EvalQuestionSchema(BaseModel):
    id:str = Field(...,pattern=r"^Q\d{3}$")
    question: str = Field(..., min_length=15, max_length=300)
    ground_truth: str = Field(..., min_length=1)
    category: Literal[
        "revenue_trends",
        "risk_disclosures",
        "segment_commentary",
        "management_guidance"
        ]
    expected_source: Literal["10-K"]
    target_companies: list[str] = Field(..., min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    requires_table_parsing: bool
    reference_period: str

    @field_validator("question")
    @classmethod
    def must_end_with_questionmark(cls,v):
        if not v.strip().endswith("?"):
            raise ValueError("question must end with '?'")
        return v
    
    @field_validator("target_companies")
    @classmethod
    def target_company_must_not_be_empty(cls,v):
        for c in v:
            if not c.strip():
                raise ValueError("Company name cannot be blank.")
        return v


class QueryRequest(BaseModel):
    question:str = Field(..., min_length=3)
    top_k:int = Field(default=5,ge=1,le=20)