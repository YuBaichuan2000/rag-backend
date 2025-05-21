# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from .config import settings
from .db.mongodb import init_database
from .api import chat, documents
from .vector_store.faiss_store import get_vector_store  # Import the FAISS store

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and vector store
@app.on_event("startup")
async def startup():
    # Initialize MongoDB
    app.mongodb = init_database()
    
    # Initialize FAISS vector store
    vector_store = get_vector_store()
    
    # Load existing FAISS index if available
    if os.path.exists("faiss_index"):
        try:
            vector_store.load_local("faiss_index")
            print("Loaded existing FAISS index")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("Starting with a new FAISS index")

# Include routers
app.include_router(chat.router, tags=["chat"])
app.include_router(documents.router, tags=["documents"])

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        reload=settings.DEBUG
    )