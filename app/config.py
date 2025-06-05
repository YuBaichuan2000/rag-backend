# app/config.py - ENHANCED FOR ATLAS
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import urllib.parse

# Load .env file
load_dotenv(override=True)

class Settings(BaseSettings):
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8001"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Test settings
    API_URL: str = os.getenv("API_URL", "http://localhost:8001")
    TEST_USER_ID: str = os.getenv("TEST_USER_ID", "test-user-123")
    
    # MongoDB Atlas settings - ENHANCED
    MONGODB_CONNECTION_STRING: str = os.getenv("MONGODB_CONNECTION_STRING", "")
    DB_NAME: str = os.getenv("DB_NAME", "prenatal_chatbot")
    
    # Collection names
    DOCUMENTS_COLLECTION: str = os.getenv("DOCUMENTS_COLLECTION", "documents")
    VECTORS_COLLECTION: str = os.getenv("VECTORS_COLLECTION", "vectors")
    CHAT_HISTORY_COLLECTION: str = os.getenv("CHAT_HISTORY_COLLECTION", "conversations")
    MESSAGES_COLLECTION: str = os.getenv("MESSAGES_COLLECTION", "messages")
    # LangGraph MongoDB Checkpointer Settings
    LANGGRAPH_CHECKPOINT_COLLECTION: str = os.getenv("LANGGRAPH_CHECKPOINT_COLLECTION", "langgraph_checkpoints")
    ENABLE_MONGODB_CHECKPOINTER: bool = os.getenv("ENABLE_MONGODB_CHECKPOINTER", "True").lower() == "true"

    # Vector Store Settings
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "mongodb")  # "mongodb" or "faiss"
    
    # MongoDB Atlas Vector Search Settings (for production)
    ATLAS_VECTOR_INDEX_NAME: str = os.getenv("ATLAS_VECTOR_INDEX_NAME", "vector_index")
    
    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # LLM settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    
    # Document processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Vector search settings
    SIMILARITY_SEARCH_K: int = int(os.getenv("SIMILARITY_SEARCH_K", "4"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Atlas-specific settings
    ATLAS_PROJECT_ID: str = os.getenv("ATLAS_PROJECT_ID", "")
    ATLAS_CLUSTER_NAME: str = os.getenv("ATLAS_CLUSTER_NAME", "")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_settings()
    
    def validate_settings(self):
        """Validate critical settings"""
        # Check if using Atlas (contains mongodb+srv or mongodb.net)
        is_atlas = "mongodb+srv://" in self.MONGODB_CONNECTION_STRING or "mongodb.net" in self.MONGODB_CONNECTION_STRING
        
        if is_atlas:
            print("🌐 Detected MongoDB Atlas connection")
            
            # Validate connection string format
            if not self.MONGODB_CONNECTION_STRING.startswith(("mongodb://", "mongodb+srv://")):
                raise ValueError("Invalid MongoDB connection string format")
            
            # Check for credentials in Atlas connection string
            if "@" not in self.MONGODB_CONNECTION_STRING:
                print("⚠️ Warning: No credentials found in Atlas connection string")
            
            # Validate database name
            if not self.DB_NAME:
                raise ValueError("DB_NAME is required for Atlas connections")
                
            print(f"📊 Atlas Database: {self.DB_NAME}")
            print(f"📁 Collections: docs={self.DOCUMENTS_COLLECTION}, vectors={self.VECTORS_COLLECTION}")
        else:
            print("🏠 Detected local MongoDB connection")
        
        # Validate OpenAI API key
        if not self.OPENAI_API_KEY:
            print("⚠️ Warning: OPENAI_API_KEY not set - RAG functionality will not work")
        else:
            print(f"🤖 OpenAI API Key configured (model: {self.LLM_MODEL})")
    
    def get_connection_info(self):
        """Get connection information for debugging"""
        # Don't log the full connection string for security
        conn_str = self.MONGODB_CONNECTION_STRING
        if "@" in conn_str:
            # Hide credentials
            parts = conn_str.split("@")
            if len(parts) == 2:
                protocol_and_creds = parts[0]
                host_and_params = parts[1]
                # Extract protocol
                protocol = protocol_and_creds.split("://")[0]
                masked_conn_str = f"{protocol}://[CREDENTIALS_HIDDEN]@{host_and_params}"
            else:
                masked_conn_str = "[CONNECTION_STRING_HIDDEN]"
        else:
            masked_conn_str = conn_str
        
        return {
            "connection_string": masked_conn_str,
            "database": self.DB_NAME,
            "collections": {
                "documents": self.DOCUMENTS_COLLECTION,
                "vectors": self.VECTORS_COLLECTION,
                "chat_history": self.CHAT_HISTORY_COLLECTION
            },
            "is_atlas": "mongodb.net" in self.MONGODB_CONNECTION_STRING or "mongodb+srv://" in self.MONGODB_CONNECTION_STRING
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()

# Print configuration info on startup
if __name__ == "__main__" or settings.DEBUG:
    print("🔧 Configuration loaded:")
    info = settings.get_connection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")