# app/db/mongodb.py
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from ..config import settings

# Synchronous client
def get_mongodb_client():
    """Get MongoDB client"""
    return MongoClient(settings.MONGODB_CONNECTION_STRING)

# Async client
def get_async_mongodb_client():
    """Get async MongoDB client"""
    return AsyncIOMotorClient(settings.MONGODB_CONNECTION_STRING)

# Database access
def get_database():
    """Get database"""
    client = get_mongodb_client()
    return client[settings.DB_NAME]

# Async database access
async def get_async_database():
    """Get async database"""
    client = get_async_mongodb_client()
    return client[settings.DB_NAME]

# Initialize database
def init_database():
    """Initialize database and collections - Updated for unified schema"""
    db = get_database()
    
    # Create collections if they don't exist
    collections = [
        settings.CHAT_HISTORY_COLLECTION,  # Updated from CHAT_HISTORY_COLLECTION
        settings.MESSAGES_COLLECTION,       # New collection for individual messages
        settings.DOCUMENTS_COLLECTION,
        settings.VECTORS_COLLECTION
    ]
    
    for collection in collections:
        if collection not in db.list_collection_names():
            db.create_collection(collection)
    
    # Create indexes for better performance
    try:
        # Conversations indexes
        db.conversations.create_index([("conversation_id", 1)], unique=True)
        db.conversations.create_index([("user_id", 1), ("updated_at", -1)])
        
        # Messages indexes
        db.messages.create_index([("conversation_id", 1), ("timestamp", 1)])
        db.messages.create_index([("message_id", 1)], unique=True)
        
        # Documents indexes (existing)
        db.documents.create_index([("user_id", 1)])
        
        print("✅ Database indexes created successfully")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create indexes: {e}")
    
    # Setup vector index if needed (existing functionality)
    setup_vector_index(db)
    
    return db

def setup_vector_index(db):
    """Set up vector search index if it doesn't exist"""
    # In a production environment with MongoDB Atlas, you would set up
    # vector search index here. For local development, this is a placeholder.
    vectors_collection = db[settings.VECTORS_COLLECTION]
    
    # Check if index exists
    try:
        existing_indexes = list(vectors_collection.list_indexes())
        index_exists = any("vector" in idx.get("name", "") for idx in existing_indexes)
        
        if not index_exists:
            print("💡 Note: For production, set up a vector search index in MongoDB Atlas")
            print("📚 Follow MongoDB Atlas documentation for vector search setup")
    except Exception as e:
        print(f"⚠️ Could not check vector indexes: {e}")