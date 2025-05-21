# app/models/api_models.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    user_id: str

class ChatResponse(BaseModel):
    response: str
    thread_id: str

class DocumentUploadResponse(BaseModel):
    document_id: str
    title: str
    status: str
    message: str

class URLUploadRequest(BaseModel):
    url: str
    title: Optional[str] = None
    user_id: str

class ConversationListResponse(BaseModel):
    conversations: List[Dict[str, Any]]