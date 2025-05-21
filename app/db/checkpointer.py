# app/db/checkpointer.py
from datetime import datetime
from langgraph.checkpoint.base import BaseCheckpointer
from .mongodb import get_database
from ..config import settings

class MongoDBCheckpointer(BaseCheckpointer):
    """Custom MongoDB checkpointer for LangGraph"""
    
    def __init__(self):
        self.db = get_database()
        self.collection = self.db[settings.CHAT_HISTORY_COLLECTION]
    
    def get(self, thread_id):
        """Load the state for a thread_id"""
        doc = self.collection.find_one({"thread_id": thread_id})
        if doc and "state" in doc:
            return doc["state"]
        return None
    
    def put(self, thread_id, state):
        """Save the state for a thread_id"""
        self.collection.update_one(
            {"thread_id": thread_id},
            {"$set": {"state": state, "updated_at": datetime.now()}},
            upsert=True
        )

    def list(self):
        """List all thread_ids"""
        return [doc["thread_id"] for doc in self.collection.find({}, {"thread_id": 1})]