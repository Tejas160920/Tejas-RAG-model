---
title: Tejas Portfolio RAG Chatbot
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# Tejas Portfolio RAG Chatbot API

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about Tejas Gaikwad based on his portfolio data.

## API Endpoints

### Health Check
- `GET /` - Returns API status
- `GET /health` - Detailed health check

### Chat
- `POST /chat` - Send a question and get a response

**Request Body:**
```json
{
  "question": "What are Tejas's skills?"
}
```

**Response:**
```json
{
  "answer": "Tejas is skilled in...",
  "success": true
}
```

## Technologies Used
- FastAPI
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS for vector search
- Groq LLM (Llama 3.3 70B)
- PDFPlumber for document processing
