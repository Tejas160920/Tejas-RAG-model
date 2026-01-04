from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from functools import lru_cache
import hashlib
import asyncio
import os
import time

# Initialize FastAPI app
app = FastAPI(title="Tejas Portfolio RAG API")

# Enable CORS for your portfolio website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []

class QueryResponse(BaseModel):
    answer: str
    success: bool

# Global variables for RAG components
embedder = None
index = None
document_keys = []
document_texts = []
llm = None

# ============================================
# CACHING LAYER - Speeds up repeated queries
# ============================================
query_cache = {}  # Cache for full responses
embedding_cache = {}  # Cache for query embeddings
CACHE_TTL = 3600  # Cache for 1 hour

def normalize_query(query: str) -> str:
    """Normalize query for better cache hits"""
    return query.lower().strip()

def get_cache_key(query: str, history_len: int) -> str:
    """Generate cache key from query and history length"""
    normalized = normalize_query(query)
    return hashlib.md5(f"{normalized}:{history_len}".encode()).hexdigest()

def get_cached_response(cache_key: str) -> Optional[str]:
    """Get cached response if exists and not expired"""
    if cache_key in query_cache:
        cached, timestamp = query_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return cached
        del query_cache[cache_key]
    return None

def set_cached_response(cache_key: str, response: str):
    """Cache a response"""
    # Keep cache size manageable (max 100 entries)
    if len(query_cache) > 100:
        oldest_key = min(query_cache.keys(), key=lambda k: query_cache[k][1])
        del query_cache[oldest_key]
    query_cache[cache_key] = (response, time.time())

def get_cached_embedding(query: str):
    """Get cached embedding for query"""
    normalized = normalize_query(query)
    if normalized in embedding_cache:
        emb, timestamp = embedding_cache[normalized]
        if time.time() - timestamp < CACHE_TTL:
            return emb
        del embedding_cache[normalized]
    return None

def set_cached_embedding(query: str, embedding):
    """Cache query embedding"""
    normalized = normalize_query(query)
    if len(embedding_cache) > 200:
        oldest_key = min(embedding_cache.keys(), key=lambda k: embedding_cache[k][1])
        del embedding_cache[oldest_key]
    embedding_cache[normalized] = (embedding, time.time())

# ============================================
# PDF & TEXT PROCESSING
# ============================================
def load_pdf_text(pdf_path):
    """Extract text from PDF"""
    pdf_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n"
    return pdf_text

def structure_text(text):
    """Process text into structured sections"""
    sections = text.split("\n# ")
    documents = {}
    for section in sections:
        lines = section.split("\n")
        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        if title and content:
            documents[title] = content
    return documents

# ============================================
# RAG RETRIEVAL WITH CACHING
# ============================================
def get_query_embedding(query: str):
    """Get embedding with caching"""
    global embedder

    cached = get_cached_embedding(query)
    if cached is not None:
        return cached

    embedding = embedder.encode(query, convert_to_numpy=True)
    set_cached_embedding(query, embedding)
    return embedding

def retrieve_context(query: str, top_k: int = 5) -> str:
    """Retrieve relevant context from FAISS index"""
    global index, document_keys, document_texts

    query_embedding = get_query_embedding(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)

    retrieved_sections = [document_keys[i] + ":\n" + document_texts[i] for i in indices[0]]
    context = "\n\n".join(retrieved_sections)

    MAX_CONTEXT_LENGTH = 5000
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]

    return context

def build_prompt(query: str, context: str, history: List[Message] = []) -> str:
    """Build the RAG prompt"""
    history_str = ""
    if history:
        recent_history = history[-6:]
        for msg in recent_history:
            role = "User" if msg.role == "user" else "Assistant"
            history_str += f"{role}: {msg.content}\n"
        history_str = f"\nCONVERSATION HISTORY:\n{history_str}\n"

    prompt = f"""You are Tejas's personal AI assistant embedded on his portfolio website. Your role is to help visitors learn about Tejas Gaikwad - his background, skills, projects, education, and experience.

PERSONALITY:
- Be friendly, professional, and conversational
- Speak as if you know Tejas well (use "Tejas" or "he" naturally)
- Be enthusiastic about his work and achievements
- Keep responses concise but informative

FORMATTING RULES (VERY IMPORTANT):
- Use bullet points (•) to list items - NEVER write long paragraphs
- Use relevant emojis to make responses engaging (e.g., 🚀 for projects, 💼 for work, 🎓 for education, 💻 for skills, 📧 for contact)
- Keep each bullet point short and scannable
- Start with a brief 1-line intro, then use bullets for details
- Example format:
  "Here are Tejas's key projects! 🚀
  • Agribot - AI-powered crop disease detection using YOLOv7
  • Virtual Try-On - Chrome extension for virtual clothing
  • Voice Cloning - 80% accuracy voice synthesis app"

INFORMATION ABOUT TEJAS:
{context}
{history_str}
RESPONSE GUIDELINES:
1. If the question is about Tejas and you have the information: Answer with a short intro + bullet points with emojis.

2. If the question is about Tejas but the specific detail isn't available: Say something like "That specific information isn't in my knowledge base, but I can tell you about [related topic]. Would you like to know more about his [projects/skills/experience/education]?"

3. If asked about contact/hiring: Direct them to his email (tejasgaikwad16092002@gmail.com) or LinkedIn (https://www.linkedin.com/in/tejasgg)

4. If asked something completely unrelated to Tejas: Politely redirect by saying "I'm Tejas's portfolio assistant, so I'm best at answering questions about him! I can tell you about his projects, skills, experience, or education. What would you like to know?"

5. For greetings: Respond warmly and offer to help them learn about Tejas.

6. IMPORTANT: If the user says "yes", "sure", "okay", "tell me more", etc., look at the conversation history to understand what they're referring to and provide that information. Don't ask again what they want - just give them the relevant details.

CURRENT QUESTION: {query}

ANSWER:"""
    return prompt

# ============================================
# STARTUP
# ============================================
@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup"""
    global embedder, index, document_keys, document_texts, llm

    print("Loading RAG components...")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set!")

    llm = ChatGroq(
        temperature=0.3,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

    pdf_file_path = "DATA.pdf"
    pdf_text = load_pdf_text(pdf_file_path)
    structured_data = structure_text(pdf_text)

    print(f"Structured sections extracted: {len(structured_data)}")

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    document_keys = list(structured_data.keys())
    document_texts = [structured_data[key] for key in document_keys]
    doc_embeddings = embedder.encode(document_texts, convert_to_numpy=True)

    embedding_dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(doc_embeddings)

    print(f"FAISS index built with {index.ntotal} documents.")
    print("RAG components loaded successfully!")

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Tejas Portfolio RAG API is running!"}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Main chat endpoint with caching"""
    global llm

    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Check cache first (only for queries without history for consistency)
        cache_key = get_cache_key(request.question, len(request.history or []))
        if not request.history:
            cached = get_cached_response(cache_key)
            if cached:
                print(f"Cache hit for: {request.question[:50]}...")
                return QueryResponse(answer=cached, success=True)

        # Retrieve context
        context = retrieve_context(request.question)

        # Build prompt
        prompt = build_prompt(request.question, context, request.history or [])

        # Get response from LLM (run in thread to not block)
        response = await asyncio.to_thread(llm.invoke, prompt)

        # Cache the response (only for queries without history)
        if not request.history:
            set_cached_response(cache_key, response.content)

        return QueryResponse(answer=response.content, success=True)

    except Exception as e:
        print(f"Error: {str(e)}")
        return QueryResponse(
            answer="Sorry, I encountered an error processing your question. Please try again!",
            success=False
        )

@app.post("/chat/stream")
async def chat_stream(request: QueryRequest):
    """Streaming chat endpoint - returns response word by word"""
    global llm

    async def generate():
        try:
            if not request.question.strip():
                yield "data: {\"error\": \"Question cannot be empty\"}\n\n"
                return

            # Check cache first
            cache_key = get_cache_key(request.question, len(request.history or []))
            if not request.history:
                cached = get_cached_response(cache_key)
                if cached:
                    # Stream cached response
                    words = cached.split(' ')
                    for i, word in enumerate(words):
                        yield f"data: {word}{' ' if i < len(words)-1 else ''}\n\n"
                        await asyncio.sleep(0.02)  # Small delay for smooth streaming
                    yield "data: [DONE]\n\n"
                    return

            # Retrieve context
            context = retrieve_context(request.question)
            prompt = build_prompt(request.question, context, request.history or [])

            # Get response and stream it
            response = await asyncio.to_thread(llm.invoke, prompt)
            full_response = response.content

            # Cache response
            if not request.history:
                set_cached_response(cache_key, full_response)

            # Stream word by word
            words = full_response.split(' ')
            for i, word in enumerate(words):
                yield f"data: {word}{' ' if i < len(words)-1 else ''}\n\n"
                await asyncio.sleep(0.02)

            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "embedder_loaded": embedder is not None,
        "index_loaded": index is not None,
        "llm_loaded": llm is not None,
        "documents_count": len(document_keys),
        "cache_size": len(query_cache),
        "embedding_cache_size": len(embedding_cache)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
