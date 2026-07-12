from pydantic import BaseModel,Field, decorator, field_validator
from typing import Literal
import uuid

class SourceSchema(BaseModel):
    chunk_id: int
    source: str
    ticker: str
    year: str
    preview: str

class AnswerSchema(BaseModel):
    answer:str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory = list)
    sources: list[SourceSchema] = Field(default_factory= list)
    prompt_tokens: int = Field(ge=0.0)
    completion_tokens: int = Field(ge=0.0)
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
    sources: Literal["10-K"]
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
    db:str = "weaviate" #default
    thread_id:str = Field(default_factory=lambda: str(uuid.uuid4()))