from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import os

# Initialize FastAPI app
app = FastAPI(title="Tejas Portfolio RAG API")

# Enable CORS for your portfolio website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class Message(BaseModel):
    role: str  # "user" or "assistant"
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

def build_prompt(query, history=[], top_k=5):
    """Build the RAG prompt with retrieved context and conversation history"""
    global embedder, index, document_keys, document_texts

    query_embedding = embedder.encode(query, convert_to_numpy=True)
    distances, indices = index.search(np.array([query_embedding]), top_k)

    retrieved_sections = [document_keys[i] + ":\n" + document_texts[i] for i in indices[0]]
    context = "\n\n".join(retrieved_sections)

    MAX_CONTEXT_LENGTH = 5000
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]

    # Build conversation history string (last 6 messages max)
    history_str = ""
    if history:
        recent_history = history[-6:]  # Keep last 6 messages for context
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

@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup"""
    global embedder, index, document_keys, document_texts, llm

    print("Loading RAG components...")

    # Initialize Groq LLM - API key must be set as environment variable in HuggingFace Spaces
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set!")
    llm = ChatGroq(
        temperature=0.3,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

    # Load and process PDF
    pdf_file_path = "DATA.pdf"
    pdf_text = load_pdf_text(pdf_file_path)
    structured_data = structure_text(pdf_text)

    print(f"Structured sections extracted: {len(structured_data)}")

    # Initialize embedder and FAISS index
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    document_keys = list(structured_data.keys())
    document_texts = [structured_data[key] for key in document_keys]
    doc_embeddings = embedder.encode(document_texts, convert_to_numpy=True)

    embedding_dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(doc_embeddings)

    print(f"FAISS index built with {index.ntotal} documents.")
    print("RAG components loaded successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Tejas Portfolio RAG API is running!"}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Main chat endpoint"""
    global llm

    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Build prompt with retrieved context and history
        prompt = build_prompt(request.question, request.history or [], top_k=5)

        # Get response from LLM
        response = llm.invoke(prompt)

        return QueryResponse(
            answer=response.content,
            success=True
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return QueryResponse(
            answer="Sorry, I encountered an error processing your question. Please try again!",
            success=False
        )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "embedder_loaded": embedder is not None,
        "index_loaded": index is not None,
        "llm_loaded": llm is not None,
        "documents_count": len(document_keys)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
