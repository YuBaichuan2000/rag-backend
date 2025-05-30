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
    """Process a chat message and return a response - Updated for unified schema"""
    try:
        # Get RAG engine
        rag_engine = get_rag_engine()
        
        # Process the message
        response_text, thread_id = rag_engine.process_message(
            request.message, 
            request.thread_id
        )
        
        # # Get database
        # db = get_database()
        
        # # If this is a new conversation, create conversation record
        # if not request.thread_id:
        #     # Create new conversation record (aligned with Express schema)
        #     conversation_doc = {
        #         "conversation_id": thread_id,
        #         "user_id": request.user_id,
        #         "title": request.message[:50] + ("..." if len(request.message) > 50 else ""),
        #         "created_at": datetime.now(),
        #         "updated_at": datetime.now(),
        #         "message_count": 0,
        #         "last_message_preview": ""
        #     }
        #     db.conversations.insert_one(conversation_doc)
        
        # # Save user message to messages collection
        # user_message_id = str(uuid.uuid4())
        # user_message = {
        #     "conversation_id": thread_id,
        #     "message_id": user_message_id,
        #     "type": "user",
        #     "content": request.message,
        #     "timestamp": datetime.now(),
        #     "metadata": {"user_id": request.user_id}
        # }
        # db.messages.insert_one(user_message)
        
        # # Save AI response to messages collection
        # ai_message_id = str(uuid.uuid4())
        # ai_message = {
        #     "conversation_id": thread_id,
        #     "message_id": ai_message_id,
        #     "type": "ai",
        #     "content": response_text,
        #     "timestamp": datetime.now(),
        #     "metadata": {
        #         "model_used": settings.LLM_MODEL,
        #         "temperature": settings.LLM_TEMPERATURE
        #     }
        # }
        # db.messages.insert_one(ai_message)
        
        # # Update conversation metadata
        # db.conversations.update_one(
        #     {"conversation_id": thread_id},
        #     {
        #         "$set": {
        #             "updated_at": datetime.now(),
        #             "last_message_preview": response_text[:100] + ("..." if len(response_text) > 100 else "")
        #         },
        #         "$inc": {"message_count": 2}  # User + AI message
        #     }
        # )
        
        return ChatResponse(
            response=response_text, 
            thread_id=thread_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(user_id: str):
    """List all conversations for a user - Updated to use conversations collection"""
    try:
        # Get database
        db = get_database()
        
        # Find all conversations for this user (aligned with Express)
        conversations = list(db.conversations.find(
            {"user_id": user_id},
            {"_id": 0}  # Exclude MongoDB _id field
        ).sort("updated_at", -1).limit(50))
        
        # Format the response
        return ConversationListResponse(conversations=conversations)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")

@router.post("/new-conversation", response_model=ChatResponse)
async def new_conversation(user_id: str):
    """Start a new conversation - Updated for unified schema"""
    try:
        # Generate a new conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Store initial conversation record
        db = get_database()
        conversation_doc = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": "New Conversation",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "message_count": 0,
            "last_message_preview": ""
        }
        db.conversations.insert_one(conversation_doc)
        
        return ChatResponse(
            response="I'm ready to help you with any questions about your pregnancy journey!",
            thread_id=conversation_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating new conversation: {str(e)}")

# New endpoint to get conversation messages (matching Express API)
@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """Get messages for a specific conversation"""
    try:
        db = get_database()
        
        # Get conversation info
        conversation = db.conversations.find_one(
            {"conversation_id": conversation_id},
            {"_id": 0}
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages for this conversation
        messages = list(db.messages.find(
            {"conversation_id": conversation_id},
            {"_id": 0}  # Exclude MongoDB _id field
        ).sort("timestamp", 1))
        
        return {
            "conversation": conversation,
            "messages": messages,
            "total_messages": len(messages)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversation messages: {str(e)}")

# Delete conversation endpoint (matching Express API)
@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: str):
    """Delete a conversation and all its messages"""
    try:
        db = get_database()
        
        # Verify conversation exists and belongs to user
        conversation = db.conversations.find_one({
            "conversation_id": conversation_id,
            "user_id": user_id
        })
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found or unauthorized")
        
        # Delete all messages in the conversation
        db.messages.delete_many({"conversation_id": conversation_id})
        
        # Delete the conversation
        db.conversations.delete_one({"conversation_id": conversation_id})
        
        return {"message": "Conversation deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")