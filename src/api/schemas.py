from pydantic import BaseModel,Field

class AnswerSchema(BaseModel):
    answer:str
    confidenece: float = Field(ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory = list)
    sources: list[str] = Field(default_factory= list)
    cost_usd: float = Field(ge=0.0)