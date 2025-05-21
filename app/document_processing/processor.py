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
    print(f"\n==== DOCUMENT PROCESSING ====")
    print(f"Processing {len(documents)} documents for user {user_id}")
    
    try:
        print(f"Step 1: Getting database connection")
        db = get_database()
        print(f"✅ Database connection successful")
        
        # Split documents into chunks
        print(f"Step 2: Splitting documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks")
        
        # Store original documents in MongoDB
        print(f"Step 3: Storing original documents in MongoDB")
        docs_collection = db[settings.DOCUMENTS_COLLECTION]
        document_ids = []
        
        for i, doc in enumerate(documents):
            doc_id = str(uuid.uuid4())
            document_ids.append(doc_id)
            print(f"  Storing document {i+1} with ID: {doc_id}")
            
            # Store document metadata and content
            doc_record = {
                "_id": doc_id,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "user_id": user_id,
                "date_added": datetime.now()
            }
            try:
                result = docs_collection.insert_one(doc_record)
                print(f"  ✅ MongoDB insert successful: {result.acknowledged}")
            except Exception as e:
                print(f"  ❌ MongoDB insert error: {str(e)}")
                raise
            
            # Add document_id to metadata for reference
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["document_id"] = doc_id
        
        # Add parent document reference to chunks
        print(f"Step 4: Adding parent document references to chunks")
        for i, chunk in enumerate(chunks):
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
                
            if i < 3 or i == len(chunks) - 1:
                print(f"  Chunk {i+1}/{len(chunks)} metadata: {chunk.metadata}")
        
        # Add to FAISS vector store
        print(f"Step 5: Adding chunks to FAISS vector store")
        try:
            vector_store = get_vector_store()
            print(f"  Vector store initialized")
            
            # Check OpenAI API key
            openai_key = settings.OPENAI_API_KEY
            if not openai_key:
                print(f"  ❌ WARNING: OPENAI_API_KEY is not set")
            else:
                print(f"  ✅ OPENAI_API_KEY is set (length: {len(openai_key)})")
                
            vector_store.add_documents(chunks, user_id)
            print(f"  ✅ Successfully added chunks to vector store")
            
            # Optionally save the FAISS index for persistence
            vector_store.save_local("faiss_index")
            print(f"  ✅ FAISS index saved")
        except Exception as e:
            import traceback
            print(f"  ❌ Vector store error: {str(e)}")
            print(f"  ❌ Traceback: {traceback.format_exc()}")
            raise
        
        print(f"✅ Successfully processed documents: {document_ids}")
        return document_ids
    
    except Exception as e:
        import traceback
        print(f"❌ ERROR in process_and_store_documents: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")