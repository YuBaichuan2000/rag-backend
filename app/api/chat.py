# app/api/chat.py
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends

from ..models.api_models import ChatRequest, ChatResponse, ConversationListResponse
from ..rag.engine import get_rag_engine
from ..db.mongodb import get_database
from ..config import settings

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a chat message and return a response"""
    try:
        # Get RAG engine
        rag_engine = get_rag_engine()
        
        # Process the message
        response_text, thread_id = rag_engine.process_message(
            request.message, 
            request.thread_id
        )
        
        # Update chat history with user info
        db = get_database()
        db[settings.CHAT_HISTORY_COLLECTION].update_one(
            {"thread_id": thread_id},
            {"$set": {
                "user_id": request.user_id, 
                "last_active": datetime.now()
            }},
            upsert=True
        )
        
        return ChatResponse(response=response_text, thread_id=thread_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(user_id: str):
    """List all conversations for a user"""
    try:
        # Get database
        db = get_database()
        
        # Find all conversations for this user
        conversations = list(db[settings.CHAT_HISTORY_COLLECTION].find(
            {"user_id": user_id},
            {"thread_id": 1, "last_active": 1, "_id": 0}
        ))
        
        # Format the response
        return ConversationListResponse(conversations=conversations)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")

@router.post("/new-conversation", response_model=ChatResponse)
async def new_conversation(user_id: str):
    """Start a new conversation"""
    try:
        # Generate a new thread ID
        thread_id = str(uuid.uuid4())
        
        # Store initial conversation record
        db = get_database()
        db[settings.CHAT_HISTORY_COLLECTION].insert_one({
            "thread_id": thread_id,
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_active": datetime.now()
        })
        
        return ChatResponse(
            response="I'm ready to help you with any questions about your documents!",
            thread_id=thread_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating new conversation: {str(e)}")