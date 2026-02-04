> #  Book Navigator RAG API

This is a **FastAPI-based backend** service for a Retrieval-Augmented Generation (RAG) system designed to act as a "Book Navigator" for the book "The Carbonated Body". It uses LangChain for RAG orchestration, OpenAI for embeddings and LLM inference, and Pinecone as the vector database. The system strictly follows a "locate, don't explain" philosophy, guiding users to relevant book sections without summarizing or explaining content.
The API exposes endpoints for querying the book (e.g., finding page locations for concepts) and includes basic security for production use. It's built to integrate with a WordPress website via API calls (e.g., from JavaScript in a custom plugin).


## Project Structure


```text
rag-api/
│
├── app/                     # Core application package
│   ├── __init__.py          # Makes 'app' a Python package
│   ├── main.py              # FastAPI app setup, routes, and middleware
│   ├── rag.py               # RAG chain logic, retriever, prompt, and utilities
│   ├── schemas.py           # Pydantic models for requests/responses
│   └── security.py          # API authentication and validation
│
├── .env                     # Environment variables (API keys – do NOT commit)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```


### Features

- RAG Pipeline: Retrieves relevant chunks from Pinecone, formats context, and uses OpenAI to generate navigation responses (e.g., page locations).

- Strict Response Rules: Responses follow a fixed structure to avoid explanations, summaries, or insights – only guiding to book sections.

- Security: Optional API key authentication via Bearer token.

- CORS Support: Configurable for WordPress integration.

- Debug Options: Optionally return retrieved context for tuning.

- Extensibility: Easy to add endpoints (e.g., for updating the vector DB – see "Future Enhancements" below).


### Setup

pip install -r requirements.txt
```text
OPENAI_API_KEY=sk-...  # Your OpenAI API key
PINECONE_API_KEY=pcsk-...  # Your Pinecone API key
API_SECRET=your-secret-key  # Optional: Secret for API authentication (recommended for production)
```

### Endpoints
The API provides the following endpoints. All responses are in JSON format.

1. GET /health


- Description: Health check endpoint to verify the service is running.

- Use Cases:

- Monitoring tools (e.g., AWS health checks) to confirm API availability.

- Quick ping from WordPress or client-side scripts to ensure the backend is responsive before querying.


