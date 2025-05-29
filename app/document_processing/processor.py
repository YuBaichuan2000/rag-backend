# app/document_processing/processor.py - Updated for MongoDB Vector Store
import uuid
from datetime import datetime
from typing import List
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from ..config import settings
from ..db.mongodb import get_database
from ..vector_store import get_vector_store  # Uses factory pattern now

async def process_and_store_documents(documents: List[Document], user_id: str):
    """Process documents and store in MongoDB with vector embeddings"""
    print(f"\n==== DOCUMENT PROCESSING (MongoDB Vector Store) ====")
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
                "date_added": datetime.now(),
                "chunk_count": 0,  # Will be updated after chunking
                "processing_status": "processing"
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
        chunk_count_by_doc = {}
        
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
                
                parent_doc_id = parent_doc.metadata["document_id"]
                chunk.metadata["parent_document_id"] = parent_doc_id
                chunk.metadata["document_id"] = parent_doc_id  # For compatibility
                chunk.metadata["user_id"] = user_id
                chunk.metadata["chunk_index"] = i
                
                # Count chunks per document
                if parent_doc_id not in chunk_count_by_doc:
                    chunk_count_by_doc[parent_doc_id] = 0
                chunk_count_by_doc[parent_doc_id] += 1
                
            if i < 3 or i == len(chunks) - 1:
                print(f"  Chunk {i+1}/{len(chunks)} metadata: {chunk.metadata}")
        
        # Add chunks to MongoDB vector store
        print(f"Step 5: Adding chunks to MongoDB vector store")
        try:
            vector_store = get_vector_store()
            print(f"  Vector store initialized: {type(vector_store).__name__}")
            
            # Check OpenAI API key
            openai_key = settings.OPENAI_API_KEY
            if not openai_key:
                print(f"  ❌ WARNING: OPENAI_API_KEY is not set")
            else:
                print(f"  ✅ OPENAI_API_KEY is set (length: {len(openai_key)})")
            
            # Add documents to vector store
            print(f"  Adding {len(chunks)} chunks to vector store...")
            vector_store.add_documents(chunks, user_id)
            print(f"  ✅ Successfully added chunks to MongoDB vector store")
            
            # Get vector store statistics
            try:
                stats = vector_store.get_stats()
                print(f"  📊 Vector store now contains {stats.get('total_documents', 0)} total documents")
            except Exception as e:
                print(f"  ⚠️ Could not get vector store stats: {e}")
                
        except Exception as e:
            import traceback
            print(f"  ❌ Vector store error: {str(e)}")
            print(f"  ❌ Traceback: {traceback.format_exc()}")
            raise
        
        # Update document records with chunk counts and completion status
        print(f"Step 6: Updating document records with processing results")
        for doc_id in document_ids:
            chunk_count = chunk_count_by_doc.get(doc_id, 0)
            try:
                docs_collection.update_one(
                    {"_id": doc_id},
                    {
                        "$set": {
                            "chunk_count": chunk_count,
                            "processing_status": "completed",
                            "processing_completed_at": datetime.now()
                        }
                    }
                )
                print(f"  ✅ Updated document {doc_id}: {chunk_count} chunks")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not update document {doc_id}: {e}")
        
        print(f"✅ Successfully processed documents: {document_ids}")
        print(f"📊 Summary:")
        print(f"  - Original documents: {len(documents)}")
        print(f"  - Generated chunks: {len(chunks)}")
        print(f"  - Document IDs: {document_ids}")
        
        return document_ids
    
    except Exception as e:
        import traceback
        print(f"❌ ERROR in process_and_store_documents: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Update any documents that were created to show error status
        try:
            if 'document_ids' in locals() and document_ids:
                db = get_database()
                docs_collection = db[settings.DOCUMENTS_COLLECTION]
                for doc_id in document_ids:
                    docs_collection.update_one(
                        {"_id": doc_id},
                        {
                            "$set": {
                                "processing_status": "error",
                                "processing_error": str(e),
                                "processing_error_at": datetime.now()
                            }
                        }
                    )
        except Exception as cleanup_error:
            print(f"⚠️ Could not update error status: {cleanup_error}")
            
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

# Additional utility functions for MongoDB vector store

async def get_document_chunks(document_id: str, user_id: str = None) -> List[Document]:
    """Retrieve all chunks for a specific document"""
    try:
        vector_store = get_vector_store()
        
        # For MongoDB vector store, we need to query by document_id
        if hasattr(vector_store, 'collection'):
            # MongoDB vector store
            filter_query = {"metadata.document_id": document_id}
            if user_id:
                filter_query["metadata.user_id"] = user_id
            
            results = list(vector_store.collection.find(filter_query).sort("metadata.chunk_index", 1))
            
            documents = []
            for result in results:
                doc = Document(
                    page_content=result["text"],
                    metadata=result["metadata"]
                )
                documents.append(doc)
            
            return documents
        else:
            # Fallback for other vector stores
            print("⚠️ get_document_chunks not fully implemented for this vector store type")
            return []
            
    except Exception as e:
        print(f"Error retrieving document chunks: {str(e)}")
        return []

async def delete_document_vectors(document_id: str, user_id: str = None) -> int:
    """Delete all vector embeddings for a specific document"""
    try:
        vector_store = get_vector_store()
        
        if hasattr(vector_store, 'delete_by_document'):
            # MongoDB vector store with delete method
            deleted_count = vector_store.delete_by_document(document_id)
            print(f"Deleted {deleted_count} vectors for document {document_id}")
            return deleted_count
        else:
            print("⚠️ delete_document_vectors not implemented for this vector store type")
            return 0
            
    except Exception as e:
        print(f"Error deleting document vectors: {str(e)}")
        raise

async def get_user_vector_stats(user_id: str) -> dict:
    """Get vector storage statistics for a user"""
    try:
        vector_store = get_vector_store()
        
        if hasattr(vector_store, 'collection'):
            # MongoDB vector store
            user_doc_count = vector_store.collection.count_documents({"metadata.user_id": user_id})
            
            # Get document breakdown
            pipeline = [
                {"$match": {"metadata.user_id": user_id}},
                {"$group": {
                    "_id": "$metadata.document_id",
                    "chunk_count": {"$sum": 1},
                    "latest_created": {"$max": "$created_at"}
                }},
                {"$sort": {"latest_created": -1}}
            ]
            
            doc_breakdown = list(vector_store.collection.aggregate(pipeline))
            
            return {
                "user_id": user_id,
                "total_chunks": user_doc_count,
                "unique_documents": len(doc_breakdown),
                "document_breakdown": doc_breakdown
            }
        else:
            return {"user_id": user_id, "error": "Stats not available for this vector store type"}
            
    except Exception as e:
        print(f"Error getting user vector stats: {str(e)}")
        return {"user_id": user_id, "error": str(e)}