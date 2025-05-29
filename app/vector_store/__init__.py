# app/vector_store/__init__.py
"""
Vector Store Factory - Choose between MongoDB and FAISS based on configuration
"""
from ..config import settings

def get_vector_store():
    """Factory function to get the appropriate vector store based on configuration"""
    
    if settings.VECTOR_STORE_TYPE == "mongodb":
        print("🔗 Using MongoDB Vector Store")
        from .mongodb_store import get_vector_store as get_mongodb_store
        return get_mongodb_store()
    
    elif settings.VECTOR_STORE_TYPE == "faiss":
        print("📁 Using FAISS Vector Store")
        from .faiss_store import get_vector_store as get_faiss_store
        return get_faiss_store()
    
    else:
        print(f"⚠️ Unknown vector store type: {settings.VECTOR_STORE_TYPE}")
        print("🔗 Falling back to MongoDB Vector Store")
        from .mongodb_store import get_vector_store as get_mongodb_store
        return get_mongodb_store()

# For backward compatibility, export both stores
from .mongodb_store import get_vector_store as get_mongodb_store
from .faiss_store import get_vector_store as get_faiss_store