# app/document_processing/processor.py - FIXED FOR ATLAS
import uuid
from datetime import datetime
from typing import List
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from ..config import settings
from ..db.mongodb import get_database
from ..vector_store.faiss_store import get_vector_store

async def process_and_store_documents(documents: List[Document], user_id: str):
    """Process documents and store in MongoDB Atlas with enhanced error handling"""
    print(f"\n==== DOCUMENT PROCESSING (ATLAS ENHANCED) ====")
    print(f"Processing {len(documents)} documents for user {user_id}")
    
    try:
        # Step 1: Test database connection first
        print(f"Step 1: Testing MongoDB Atlas connection")
        try:
            db = get_database()
            # Test the connection with a simple ping
            db.command('ping')
            print(f"✅ Atlas connection successful")
            
            # Test collection access
            docs_collection = db[settings.DOCUMENTS_COLLECTION]
            print(f"✅ Documents collection access successful: {settings.DOCUMENTS_COLLECTION}")
            
        except Exception as e:
            print(f"❌ Atlas connection failed: {str(e)}")
            # Try to provide more specific error information
            if "authentication failed" in str(e).lower():
                raise HTTPException(
                    status_code=500, 
                    detail=f"MongoDB Atlas authentication failed. Check username/password and IP whitelist."
                )
            elif "connection" in str(e).lower():
                raise HTTPException(
                    status_code=500, 
                    detail=f"Cannot connect to MongoDB Atlas. Check connection string and network access."
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Database connection error: {str(e)}"
                )
        
        # Step 2: Split documents into chunks
        print(f"Step 2: Splitting documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks")
        
        # Step 3: Store original documents in MongoDB Atlas
        print(f"Step 3: Storing original documents in Atlas")
        document_ids = []
        
        for i, doc in enumerate(documents):
            doc_id = str(uuid.uuid4())
            document_ids.append(doc_id)
            print(f"  Storing document {i+1} with ID: {doc_id}")
            
            # Prepare document for Atlas
            doc_record = {
                "_id": doc_id,  # Use our UUID as the _id
                "content": doc.page_content,
                "metadata": doc.metadata if doc.metadata else {},
                "user_id": user_id,
                "date_added": datetime.now(),
                "status": "processed",
                "chunk_count": 0  # Will be updated later
            }
            
            try:
                # Insert with proper error handling for Atlas
                result = docs_collection.insert_one(doc_record)
                print(f"  ✅ Atlas insert successful: {result.acknowledged}")
                print(f"     Inserted ID: {result.inserted_id}")
                
            except Exception as e:
                print(f"  ❌ Atlas insert error: {str(e)}")
                # Handle specific Atlas errors
                if "duplicate key" in str(e).lower():
                    print(f"  ⚠️ Document with ID {doc_id} already exists, skipping...")
                    continue
                elif "authentication" in str(e).lower():
                    raise HTTPException(
                        status_code=500, 
                        detail="Atlas authentication failed during document insert"
                    )
                else:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Failed to store document in Atlas: {str(e)}"
                    )
            
            # Add document_id to metadata for reference
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["document_id"] = doc_id
        
        # Step 4: Process chunks and add parent references
        print(f"Step 4: Processing chunks with parent references")
        chunks_processed = 0
        
        for i, chunk in enumerate(chunks):
            # Find parent document for this chunk
            parent_doc = None
            for doc in documents:
                if chunk.page_content in doc.page_content:
                    parent_doc = doc
                    break
            
            if parent_doc and "document_id" in parent_doc.metadata:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata["parent_document_id"] = parent_doc.metadata["document_id"]
                chunk.metadata["user_id"] = user_id
                chunk.metadata["chunk_index"] = i
                chunks_processed += 1
                
            if i < 3 or i == len(chunks) - 1:
                print(f"  Chunk {i+1}/{len(chunks)} metadata: {chunk.metadata}")
        
        print(f"✅ Processed {chunks_processed} chunks with parent references")
        
        # Step 5: Update document chunk counts in Atlas
        print(f"Step 5: Updating document chunk counts")
        for doc_id in document_ids:
            chunk_count = sum(1 for chunk in chunks 
                            if chunk.metadata and 
                            chunk.metadata.get("parent_document_id") == doc_id)
            
            try:
                update_result = docs_collection.update_one(
                    {"_id": doc_id},
                    {"$set": {"chunk_count": chunk_count, "last_updated": datetime.now()}}
                )
                print(f"  Updated document {doc_id}: {chunk_count} chunks")
                
            except Exception as e:
                print(f"  ⚠️ Warning: Could not update chunk count for {doc_id}: {e}")
        
        # Step 6: Add to FAISS vector store
        print(f"Step 6: Adding chunks to FAISS vector store")
        try:
            vector_store = get_vector_store()
            print(f"  Vector store initialized")
            
            # Validate OpenAI API key
            openai_key = settings.OPENAI_API_KEY
            if not openai_key:
                print(f"  ❌ ERROR: OPENAI_API_KEY is not set")
                raise HTTPException(
                    status_code=500, 
                    detail="OpenAI API key not configured. Vector embeddings cannot be created."
                )
            else:
                print(f"  ✅ OPENAI_API_KEY is configured (length: {len(openai_key)})")
            
            # Add documents to vector store
            if chunks:
                vector_store.add_documents(chunks, user_id)
                print(f"  ✅ Successfully added {len(chunks)} chunks to vector store")
                
                # Save the FAISS index
                vector_store.save_local("faiss_index")
                print(f"  ✅ FAISS index saved to disk")
            else:
                print(f"  ⚠️ No chunks to add to vector store")
            
        except Exception as e:
            print(f"  ❌ Vector store error: {str(e)}")
            import traceback
            print(f"  📋 Traceback: {traceback.format_exc()}")
            
            # Don't fail the entire process if vector store fails
            print(f"  ⚠️ Continuing without vector embeddings...")
        
        # Step 7: Final validation
        print(f"Step 7: Final validation")
        try:
            # Verify documents were stored
            stored_count = docs_collection.count_documents({"user_id": user_id})
            print(f"  📊 Total documents for user {user_id}: {stored_count}")
            
            # Verify our specific documents
            our_docs_count = docs_collection.count_documents({"_id": {"$in": document_ids}})
            print(f"  📊 Our documents stored: {our_docs_count}/{len(document_ids)}")
            
        except Exception as e:
            print(f"  ⚠️ Warning: Could not perform final validation: {e}")
        
        print(f"🎉 Successfully processed and stored {len(document_ids)} documents in Atlas!")
        print(f"📋 Document IDs: {document_ids}")
        return document_ids
    
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        print(f"❌ CRITICAL ERROR in process_and_store_documents: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"📋 Full traceback: {traceback.format_exc()}")
        
        # Provide more helpful error messages
        error_message = str(e)
        if "authentication" in error_message.lower():
            error_message = "MongoDB Atlas authentication failed. Please check your username, password, and IP whitelist settings."
        elif "connection" in error_message.lower():
            error_message = "Cannot connect to MongoDB Atlas. Please check your connection string and network settings."
        elif "timeout" in error_message.lower():
            error_message = "Connection to MongoDB Atlas timed out. Please check your network connection."
        
        raise HTTPException(
            status_code=500, 
            detail=f"Document processing failed: {error_message}"
        )