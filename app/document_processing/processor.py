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
from ..vector_store.faiss_store import get_vector_store  # Updated import

async def process_and_store_documents(documents: List[Document], user_id: str):
    """Process documents and store in MongoDB with FAISS for vectors"""
    try:
        print(f"Processing {len(documents)} documents for user {user_id}")
        
        db = get_database()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Store original documents in MongoDB
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
            
            # Add document_id to metadata for reference
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["document_id"] = doc_id
        
        # Add parent document reference to chunks
        for chunk in chunks:
            # Find document this chunk belongs to
            parent_doc = None
            for doc in documents:
                if chunk.page_content in doc.page_content:
                    parent_doc = doc
                    break
            
            if parent_doc and "document_id" in parent_doc.metadata:
                # Add reference to parent document
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata["parent_document_id"] = parent_doc.metadata["document_id"]
        
        # Add to FAISS vector store
        print("Adding chunks to FAISS vector store")
        vector_store = get_vector_store()
        vector_store.add_documents(chunks, user_id)
        
        # Optionally save the FAISS index for persistence
        vector_store.save_local("faiss_index")
        
        print(f"Successfully processed documents: {document_ids}")
        return document_ids
    
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")