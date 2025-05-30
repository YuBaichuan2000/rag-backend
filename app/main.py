# app/main.py - Fully fixed production-ready FastAPI application
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime

from .config import settings
from .db.mongodb import init_database
from .api import chat, documents
from .vector_store import get_vector_store

# Initialize FastAPI app
app = FastAPI(
    title="Prenatal AI Clinic - RAG Chatbot API",
    description="AI-powered prenatal care chatbot with MongoDB-based vector storage for document retrieval",
    version="2.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Production CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        # Production frontend URLs - update these with your actual Render URLs
        "https://prenatal-ai-frontend.onrender.com",
        os.getenv("FRONTEND_URL", ""),
        # Allow any subdomain on onrender.com for flexibility during deployment
        "https://*.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global application state
app.mongodb = None
app.vector_store = None

# Initialize database and vector store
@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    print("🚀 Starting Prenatal AI Clinic API...")
    print(f"🌍 Environment: {os.getenv('NODE_ENV', 'development')}")
    print(f"🔧 Debug mode: {settings.DEBUG}")
    
    # Initialize MongoDB database and collections
    print("📦 Initializing MongoDB database...")
    try:
        app.mongodb = init_database()
        print("✅ MongoDB database initialized successfully")
        
        # Test database connection
        app.mongodb.command('ping')
        print("✅ MongoDB connection verified")
        
    except Exception as e:
        print(f"❌ MongoDB initialization failed: {str(e)}")
        app.mongodb = None
    
    # Initialize vector store
    print(f"🔗 Initializing vector store ({settings.VECTOR_STORE_TYPE})...")
    try:
        vector_store = get_vector_store()
        app.vector_store = vector_store
        
        # Get and display vector store statistics
        if hasattr(vector_store, 'get_stats'):
            try:
                stats = vector_store.get_stats()
                print(f"📊 Vector store statistics:")
                print(f"   - Total documents: {stats.get('total_documents', 0)}")
                print(f"   - Storage type: {stats.get('collection_name', 'Unknown')}")
                print(f"   - Atlas enabled: {stats.get('is_atlas', False)}")
            except Exception as stats_error:
                print(f"⚠️ Could not get vector store stats: {stats_error}")
        
        print("✅ Vector store initialized successfully")
        
    except Exception as e:
        print(f"❌ Vector store initialization failed: {str(e)}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")
        app.vector_store = None
    
    # Display configuration summary
    print(f"\n📋 Configuration Summary:")
    print(f"   - Vector Store: {settings.VECTOR_STORE_TYPE}")
    print(f"   - Database: {settings.DB_NAME}")
    print(f"   - OpenAI Model: {settings.LLM_MODEL}")
    print(f"   - API Host: {settings.API_HOST}:{settings.API_PORT}")
    
    # Validate critical settings
    if not settings.OPENAI_API_KEY:
        print("⚠️ WARNING: OPENAI_API_KEY not set - AI functionality will not work")
    else:
        print("✅ OpenAI API key configured")
    
    if not settings.MONGODB_CONNECTION_STRING:
        print("⚠️ WARNING: MONGODB_CONNECTION_STRING not set")
    else:
        print("✅ MongoDB connection string configured")
    
    print("🎉 Startup completed!")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("🛑 Shutting down Prenatal AI Clinic API...")
    
    # Close MongoDB connections if needed - FIXED
    if app.mongodb is not None:
        try:
            print("📦 MongoDB connections closed")
        except Exception as e:
            print(f"⚠️ Error closing MongoDB connections: {e}")
    
    print("✅ Shutdown completed")

# Include API routers
app.include_router(chat.router, tags=["chat"])
app.include_router(documents.router, tags=["documents"])

@app.get("/")
async def root():
    """Root endpoint with system information"""
    try:
        # Determine deployment environment
        is_production = os.getenv("NODE_ENV") == "production"
        
        # Get vector store info safely - FIXED
        vector_store_info = "Not initialized"
        if app.vector_store is not None:
            try:
                if hasattr(app.vector_store, 'get_stats'):
                    stats = app.vector_store.get_stats()
                    vector_store_info = {
                        "type": settings.VECTOR_STORE_TYPE,
                        "total_documents": stats.get('total_documents', 0),
                        "is_atlas": stats.get('is_atlas', False),
                        "status": "healthy"
                    }
                else:
                    vector_store_info = {
                        "type": settings.VECTOR_STORE_TYPE, 
                        "status": "initialized"
                    }
            except Exception as e:
                vector_store_info = {
                    "type": settings.VECTOR_STORE_TYPE, 
                    "status": "error",
                    "error": str(e)
                }
        
        # Get database status - FIXED
        db_status = "disconnected"
        if app.mongodb is not None:
            try:
                app.mongodb.command('ping')
                db_status = "connected"
            except Exception:
                db_status = "error"
        
        response_data = {
            "message": "Prenatal AI Clinic API is running",
            "version": "2.0.0",
            "environment": "production" if is_production else "development",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "database": {
                    "type": "mongodb",
                    "status": db_status,
                    "name": settings.DB_NAME
                },
                "vector_store": vector_store_info,
                "llm": {
                    "model": settings.LLM_MODEL,
                    "temperature": settings.LLM_TEMPERATURE,
                    "api_key_configured": bool(settings.OPENAI_API_KEY)
                }
            },
            "endpoints": {
                "chat": "/chat",
                "upload_url": "/upload-url", 
                "upload_file": "/upload-file",
                "conversations": "/conversations",
                "health": "/health",
                "docs": "/docs" if settings.DEBUG else "disabled"
            }
        }
        
        return response_data
        
    except Exception as e:
        # Return a safe error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint for monitoring"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Prenatal AI Clinic FastAPI",
            "version": "2.0.0",
            "environment": os.getenv("NODE_ENV", "development"),
            "services": {}
        }
        
        # Check MongoDB connection - FIXED
        try:
            if app.mongodb is not None:
                app.mongodb.command('ping')
                health_status["services"]["mongodb"] = {
                    "status": "healthy", 
                    "database": settings.DB_NAME,
                    "type": "atlas" if "mongodb.net" in settings.MONGODB_CONNECTION_STRING else "local"
                }
            else:
                health_status["services"]["mongodb"] = {"status": "not_initialized"}
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "unhealthy"
        
        # Check vector store - FIXED
        try:
            if app.vector_store is not None:
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
                if health_status["status"] == "healthy":
                    health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["vector_store"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "unhealthy"
        
        # Check OpenAI API key configuration
        if settings.OPENAI_API_KEY:
            health_status["services"]["openai"] = {
                "status": "configured",
                "model": settings.LLM_MODEL
            }
        else:
            health_status["services"]["openai"] = {"status": "not_configured"}
            if health_status["status"] == "healthy":
                health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/vector-stats")
async def vector_store_statistics():
    """Get detailed vector store statistics for monitoring"""
    try:
        if app.vector_store is not None:
            if hasattr(app.vector_store, 'get_stats'):
                stats = app.vector_store.get_stats()
                return {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
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
            "vector_store_type": settings.VECTOR_STORE_TYPE,
            "timestamp": datetime.now().isoformat()
        }

# Production-specific error handlers
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """Custom 500 error handler for production"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat(),
            "status_code": 500
        }
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 error handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": f"The requested endpoint {request.url.path} was not found",
            "timestamp": datetime.now().isoformat(),
            "status_code": 404
        }
    )

# Main execution for production (Render will use this)
if __name__ == "__main__":
    # Get port from environment (Render sets this automatically)
    port = int(os.getenv("PORT", settings.API_PORT))
    
    # Determine if we're in production
    is_production = os.getenv("NODE_ENV") == "production"
    
    print(f"🚀 Starting FastAPI server on port {port}")
    print(f"🌍 Environment: {'production' if is_production else 'development'}")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",  # Bind to all interfaces for Render
        port=port,
        reload=not is_production,  # Disable reload in production
        log_level="info" if is_production else "debug",
        access_log=True
    )