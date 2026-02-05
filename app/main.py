from fastapi import FastAPI, Depends
from fastapi import HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import QueryRequest, QueryResponse
from app.rag import get_context, rag_chain
from app.security import verify_api_key

import os



app = FastAPI(
    title="Book Navigator RAG API",
    description="API for guided reading of 'The Carbonated Body'",
    version="1.0.0",
)

# CORS – restrict in production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://thecarbonatedbody.com",
        "https://www.thecarbonatedbody.com"
    ],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Get API secret from env (optional)
API_SECRET = os.getenv("API_SECRET")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "book-navigator-rag"}

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    authorization: str | None = Depends(lambda: None)  # we'll handle manually
):
    # Security check
    verify_api_key(authorization, API_SECRET)

    try:
        context = None
        if request.show_context:
            context = get_context(request.question)

        answer = rag_chain.invoke(request.question)

        return QueryResponse(
            question=request.question,
            answer=answer,
            context=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
