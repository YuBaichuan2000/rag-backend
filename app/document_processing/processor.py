# app/document_processing/processor.py
import uuid
from datetime import datetime
from typing import List
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from ..config import settings
from ..db.mongodb import get_database

async def process_and_store_documents(documents: List[Document], user_id: str):
    """Process documents and store in MongoDB"""
    try:
        db = get_database()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Store original documents
        docs_collection = db[settings.DOCUMENTS_COLLECTION]
        document_ids = []
        
        for doc in documents:
            doc_id = str(uuid.uuid4())
            document_ids.append(doc_id)
            
            # Store document metadata and content
            doc_record = {
                "_id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "user_id": user_id,
                "date_added": datetime.now()
            }
            docs_collection.insert_one(doc_record)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Add embeddings to MongoDB
        vectors_collection = db[settings.VECTORS_COLLECTION]
        
        for chunk in chunks:
            # Generate embedding
            embedding_vector = embeddings.embed_query(chunk.page_content)
            
            # Find document this chunk belongs to
            parent_doc = None
            for doc in documents:
                if chunk.page_content in doc.page_content:
                    parent_doc = doc
                    break
            
            if parent_doc:
                # Create vector record
                vector_record = {
                    "content": chunk.page_content,
                    "embedding": embedding_vector,
                    "metadata": {
                        **chunk.metadata,
                        "parent_document_id": document_ids[documents.index(parent_doc)],
                        "user_id": user_id
                    }
                }
                
                vectors_collection.insert_one(vector_record)
        
        return document_ids
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")