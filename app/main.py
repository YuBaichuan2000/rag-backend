# app/main.py - Updated for MongoDB Vector Store
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from .config import settings
from .db.mongodb import init_database
from .api import chat, documents
from .vector_store import get_vector_store  # Updated import

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API - MongoDB Vector Store",
    description="AI-powered chatbot with MongoDB-based vector storage for document retrieval",
    version="2.0.0"
)

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
    """Initialize services on startup"""
    print("🚀 Starting RAG Chatbot API with MongoDB Vector Store...")
    
    # Initialize MongoDB database and collections
    print("📦 Initializing MongoDB database...")
    app.mongodb = init_database()
    print("✅ MongoDB database initialized")
    
    # Initialize vector store (MongoDB or FAISS based on config)
    print(f"🔗 Initializing vector store ({settings.VECTOR_STORE_TYPE})...")
    try:
        vector_store = get_vector_store()
        app.vector_store = vector_store  # Store reference for access in endpoints
        
        # Get and display vector store statistics
        if hasattr(vector_store, 'get_stats'):
            stats = vector_store.get_stats()
            print(f"📊 Vector store statistics:")
            print(f"   - Total documents: {stats.get('total_documents', 0)}")
            print(f"   - Storage type: {stats.get('collection_name', 'Unknown')}")
            print(f"   - Atlas enabled: {stats.get('is_atlas', False)}")
        
        print("✅ Vector store initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing vector store: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        # Don't fail startup, but log the error
        app.vector_store = None
    
    # Display configuration summary
    print(f"\n📋 Configuration Summary:")
    print(f"   - Vector Store: {settings.VECTOR_STORE_TYPE}")
    print(f"   - MongoDB: {settings.MONGODB_CONNECTION_STRING}")
    print(f"   - Database: {settings.DB_NAME}")
    print(f"   - OpenAI Model: {settings.LLM_MODEL}")
    
    print("🎉 Startup completed successfully!")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("🛑 Shutting down RAG Chatbot API...")
    
    # Close MongoDB connections if needed
    if hasattr(app, 'mongodb'):
        # MongoDB connections are typically handled automatically
        print("📦 MongoDB connections will be closed automatically")
    
    print("✅ Shutdown completed")

# Include API routers
app.include_router(chat.router, tags=["chat"])
app.include_router(documents.router, tags=["documents"])

@app.get("/")
async def root():
    """Root endpoint with system information"""
    vector_store_info = "Not initialized"
    
    if hasattr(app, 'vector_store') and app.vector_store:
        try:
            if hasattr(app.vector_store, 'get_stats'):
                stats = app.vector_store.get_stats()
                vector_store_info = {
                    "type": settings.VECTOR_STORE_TYPE,
                    "total_documents": stats.get('total_documents', 0),
                    "is_atlas": stats.get('is_atlas', False)
                }
            else:
                vector_store_info = {"type": settings.VECTOR_STORE_TYPE, "status": "initialized"}
        except Exception as e:
            vector_store_info = {"type": settings.VECTOR_STORE_TYPE, "error": str(e)}
    
    return {
        "message": "RAG Chatbot API is running",
        "version": "2.0.0",
        "vector_store": vector_store_info,
        "database": settings.DB_NAME,
        "llm_model": settings.LLM_MODEL,
        "endpoints": {
            "chat": "/chat",
            "upload_url": "/upload-url", 
            "upload_file": "/upload-file",
            "conversations": "/conversations",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",  # Will be set dynamically
        "services": {}
    }
    
    from datetime import datetime
    health_status["timestamp"] = datetime.now().isoformat()
    
    # Check MongoDB connection
    try:
        db = init_database()
        # Simple ping to test connection
        db.command('ping')
        health_status["services"]["mongodb"] = {"status": "healthy", "database": settings.DB_NAME}
    except Exception as e:
        health_status["services"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check vector store
    try:
        if hasattr(app, 'vector_store') and app.vector_store:
            if hasattr(app.vector_store, 'get_stats'):
                stats = app.vector_store.get_stats()
                health_status["services"]["vector_store"] = {
                    "status": "healthy",
                    "type": settings.VECTOR_STORE_TYPE,
                    "documents": stats.get('total_documents', 0)
                }
            else:
                health_status["services"]["vector_store"] = {
                    "status": "healthy", 
                    "type": settings.VECTOR_STORE_TYPE
                }
        else:
            health_status["services"]["vector_store"] = {"status": "not_initialized"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["vector_store"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check OpenAI API key
    if settings.OPENAI_API_KEY:
        health_status["services"]["openai"] = {"status": "configured"}
    else:
        health_status["services"]["openai"] = {"status": "not_configured"}
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/vector-stats")
async def vector_store_statistics():
    """Get detailed vector store statistics"""
    try:
        if hasattr(app, 'vector_store') and app.vector_store:
            if hasattr(app.vector_store, 'get_stats'):
                stats = app.vector_store.get_stats()
                return {
                    "success": True,
                    "vector_store_type": settings.VECTOR_STORE_TYPE,
                    "statistics": stats
                }
            else:
                return {
                    "success": False,
                    "error": "Statistics not available for this vector store type",
                    "vector_store_type": settings.VECTOR_STORE_TYPE
                }
        else:
            return {
                "success": False,
                "error": "Vector store not initialized",
                "vector_store_type": settings.VECTOR_STORE_TYPE
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "vector_store_type": settings.VECTOR_STORE_TYPE
        }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        reload=settings.DEBUG
    )