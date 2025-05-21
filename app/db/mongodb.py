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
    """Initialize database and collections"""
    db = get_database()
    
    # Create collections if they don't exist
    collections = [
        settings.DOCUMENTS_COLLECTION,
        settings.VECTORS_COLLECTION, 
        settings.CHAT_HISTORY_COLLECTION
    ]
    
    for collection in collections:
        if collection not in db.list_collection_names():
            db.create_collection(collection)
    
    # Setup vector index if needed
    setup_vector_index(db)
    
    return db

def setup_vector_index(db):
    """Set up vector search index if it doesn't exist"""
    # In a production environment with MongoDB Atlas, you would set up
    # vector search index here. For local development, this is a placeholder.
    vectors_collection = db[settings.VECTORS_COLLECTION]
    
    # Check if index exists
    existing_indexes = vectors_collection.list_indexes()
    index_exists = any("vector" in idx.get("name", "") for idx in existing_indexes)
    
    if not index_exists:
        print("Note: For production, set up a vector search index in MongoDB Atlas")
        print("Follow MongoDB Atlas documentation for vector search setup")