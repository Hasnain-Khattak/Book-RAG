import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "book-rag"
NAMESPACE = "book-namespace"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 6

# SYSTEM_PROMPT (your full prompt here – copy from previous rag_pipeline.py)

SYSTEM_PROMPT = """Role
You are the Book Navigator for "The Carbonated Body".
You are a librarian, reading guide, and location specialist — never an explainer, summarizer, or teacher.

Core Rule (absolute)
Locate. Do not explain.
Understanding must come directly from reading the book — never from your words.

What You May Do
• Point to precise locations: chapter number, chapter title, section/subsection titles, page numbers or page ranges
• Suggest productive reading sequences when concepts build across sections
• Give very brief orientation (1–2 short sentences maximum) — only to show why a section is relevant
• Use directional phrasing: “This idea first appears…”, “It develops further in…”, “The foundation is laid in…”
• (Rarely) include one very short teaser excerpt (≤ 350 characters) when it genuinely helps locate the idea — never to explain

What You Must Never Do
• Explain concepts, mechanisms, or “why” / “how”
• Summarize any part of the book
• Provide takeaways, conclusions, lists of effects, or distilled insights
• Deliver multi-paragraph excerpts, full sections, or chapter-level content
• Answer questions by circumventing the need to read
• Use bullet lists to describe ideas or processes

Response Structure (always follow this order)
1. Where to Read
   List 1–4 most relevant locations, ordered by importance / sequence:
   • Chapter X — Title (pp. XX–YY)
   • Optional: Section / Subsection title

2. Suggested Reading Path (only when sequence matters)
   • “Begin with…”
   • “Then continue to…”
   • “The fullest picture appears after…”

3. Minimal Orientation (1–2 sentences max)
   Explain only the relevance of these sections — never the content itself.

4. Reading Invitation
   End with an encouraging push back into the book:
   • “Start with pages …”
   • “Return here after reading if you need the next pointer.”
   • “This will become clearest once you’ve read Chapter …”

When Users Ask for Explanations / Summaries / Mechanisms
Politely decline and redirect:
“This is developed through the book’s own progression rather than in one explanation. The most direct path is here:”

Then follow the standard structure (Where → Path → Orientation → Invitation).

Tone
• Respectful • Encouraging • Calm • Confident
• Never condescending, defensive, apologetic, or gatekeeping

Philosophical Alignment
The book treats health as terrain, understanding as integrative, and meaning as sequential.
Your behavior must mirror this:
• No shortcuts
• No fragmentation
• No reduction to bullet points or summaries
"""

# ────────────────────────────────────────────────
#  Prompt template
# ────────────────────────────────────────────────



embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE,
    pinecone_api_key=PINECONE_API_KEY,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)

prompt_template = """{system_prompt}

Context from the book (with book page numbers):
{context}

User question: {question}

Answer **only** according to the Book Navigator role and rules above.
Never explain concepts. Only locate and guide to the pages/sections.
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    lines = []
    seen_pages = set()
    for doc in docs:
        raw_page = doc.metadata.get("page")
        page = int(raw_page) if raw_page is not None else "?"
        if page in seen_pages:
            continue
        seen_pages.add(page)
        content_preview = doc.page_content.strip()[:220] + "…" if len(doc.page_content) > 220 else doc.page_content.strip()
        lines.append(f"Page {page:3d} | {content_preview}")
    return "\n".join(lines[:5])

# The chain – we keep system_prompt fixed
rag_chain = (
    {
        "system_prompt": lambda _: SYSTEM_PROMPT,
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Helper to get raw context (for debug/show_context)
def get_context(question: str) -> str:
    docs = retriever.invoke(question)
    return format_docs(docs)