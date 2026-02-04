from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    question: str
    show_context: Optional[bool] = False

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: Optional[str] = None