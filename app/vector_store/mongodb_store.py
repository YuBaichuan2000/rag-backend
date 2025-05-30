# app/vector_store/mongodb_store.py
import os
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from ..config import settings
from ..db.mongodb import get_database

class MongoDBVectorStore:
    """MongoDB-backed vector store for document retrieval"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.db = get_database()
        self.collection = self.db.vectors
        self._initialize_collection()
        print("Initialized MongoDB vector store")
        
    def _initialize_collection(self):
        """Initialize the vectors collection with proper indexes"""
        try:
            # Create indexes for better performance
            # Text search index for metadata
            self.collection.create_index([("metadata.user_id", 1)])
            self.collection.create_index([("metadata.document_id", 1)])
            self.collection.create_index([("created_at", -1)])
            
            # For MongoDB Atlas, you would create a vector search index here
            

            # For local MongoDB, we'll use cosine similarity calculation
            print("✅ Vector collection indexes created")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create vector indexes: {e}")
    
    def add_documents(self, documents: List[Document], user_id: Optional[str] = None):
        """Add documents to the MongoDB vector store"""
        if not documents:
            print("No documents to add")
            return
            
        print(f"Adding {len(documents)} documents to MongoDB vector store")
        print(f"User ID: {user_id}")
        
        try:
            # Generate embeddings for all documents
            texts = [doc.page_content for doc in documents]
            print(f"Generating embeddings for {len(texts)} documents...")
            
            # Get embeddings from OpenAI
            embeddings_list = self.embeddings.embed_documents(texts)
            print(f"✅ Generated {len(embeddings_list)} embeddings")
            
            # Prepare documents for insertion
            vector_docs = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings_list)):
                # Ensure metadata exists
                if not doc.metadata:
                    doc.metadata = {}
                
                # Add user_id to metadata if provided
                if user_id:
                    doc.metadata["user_id"] = user_id
                
                vector_doc = {
                    "_id": str(uuid.uuid4()),
                    "text": doc.page_content,
                    "embedding": embedding,  # Store as list for MongoDB
                    "metadata": doc.metadata,
                    "created_at": datetime.now(),
                    "embedding_model": "text-embedding-ada-002",  # Track which model was used
                    "text_length": len(doc.page_content)
                }
                vector_docs.append(vector_doc)
                
                if i < 3 or i == len(documents) - 1:
                    print(f"  Document {i+1}/{len(documents)} - Length: {len(doc.page_content)}, Metadata: {doc.metadata}")
            
            # Insert documents into MongoDB
            result = self.collection.insert_many(vector_docs)
            print(f"✅ Successfully inserted {len(result.inserted_ids)} vector documents")
            
            # Log collection stats
            total_docs = self.collection.count_documents({})
            print(f"📊 Total vectors in collection: {total_docs}")
            
        except Exception as e:
            import traceback
            print(f"❌ Error adding documents to MongoDB vector store: {str(e)}")
            print(f"❌ Traceback: {traceback.format_exc()}")
            raise
    
    def similarity_search(self, query: str, k: int = 4, user_id: Optional[str] = None) -> List[Document]:
        """Search for similar documents using MongoDB"""
        print(f"Searching MongoDB for: '{query}' (k={k}, user_id={user_id})")
        
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            print(f"✅ Generated query embedding")
            
            # Build MongoDB aggregation pipeline
            pipeline = []
            
            # Match stage - filter by user_id if provided
            match_stage = {}
            if user_id:
                match_stage["metadata.user_id"] = user_id
            
            if match_stage:
                pipeline.append({"$match": match_stage})
            
            # Add vector similarity stage
            # For MongoDB Atlas, you would use $vectorSearch here
            # For local MongoDB, we'll use a custom similarity calculation
            if self._is_atlas_available():
                # MongoDB Atlas Vector Search
                vector_search_stage = {
                    "$vectorSearch": {
                        "index": "vector_index",  # You need to create this in Atlas
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,
                        "limit": k
                    }
                }
                pipeline.insert(0, vector_search_stage)  # Insert at beginning
            else:
                # Local MongoDB: Add all documents and calculate similarity in Python
                print("Using local similarity calculation (not optimized for large datasets)")
            
            # Execute query
            if self._is_atlas_available():
                # Use MongoDB Atlas vector search
                results = list(self.collection.aggregate(pipeline))
            else:
                # Fallback: Get all documents and calculate similarity locally
                results = self._local_similarity_search(query_embedding, k, user_id)
            
            # Convert results to LangChain Documents
            documents = []
            for result in results:
                # Extract similarity score if available
                score = result.get("score", 0.0)
                
                doc = Document(
                    page_content=result["text"],
                    metadata={
                        **result["metadata"],
                        "similarity_score": score,
                        "_id": result["_id"]
                    }
                )
                documents.append(doc)
            
            print(f"Found {len(documents)} similar documents")
            if documents:
                print(f"Top result preview: {documents[0].page_content[:100]}...")
            
            return documents
            
        except Exception as e:
            print(f"❌ Error in similarity_search: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def _is_atlas_available(self) -> bool:
        """Check if we're using MongoDB Atlas with vector search capabilities"""
        try:
            # Simple check: if connection string contains mongodb.net, assume Atlas
            conn_str = settings.MONGODB_CONNECTION_STRING
            return "mongodb.net" in conn_str or "mongodb+srv" in conn_str
        except:
            return False
    
    def _local_similarity_search(self, query_embedding: List[float], k: int, user_id: Optional[str] = None) -> List[Dict]:
        """Fallback similarity search for local MongoDB without vector search"""
        # Build filter
        filter_query = {}
        if user_id:
            filter_query["metadata.user_id"] = user_id
        
        # Get all documents (not efficient for large datasets)
        all_docs = list(self.collection.find(filter_query))
        
        if not all_docs:
            return []
        
        # Calculate cosine similarity
        similarities = []
        query_vector = np.array(query_embedding)
        
        for doc in all_docs:
            doc_vector = np.array(doc["embedding"])
            
            # Cosine similarity
            dot_product = np.dot(query_vector, doc_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc_vector)
            
            if norm_query == 0 or norm_doc == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_query * norm_doc)
            
            similarities.append((doc, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:k]
        
        # Format results
        results = []
        for doc, score in top_results:
            doc["score"] = float(score)
            results.append(doc)
        
        return results
    
    def delete_by_user(self, user_id: str):
        """Delete all vectors for a specific user"""
        try:
            result = self.collection.delete_many({"metadata.user_id": user_id})
            print(f"Deleted {result.deleted_count} vectors for user {user_id}")
            return result.deleted_count
        except Exception as e:
            print(f"Error deleting vectors for user {user_id}: {str(e)}")
            raise
    
    def delete_by_document(self, document_id: str):
        """Delete all vectors for a specific document"""
        try:
            result = self.collection.delete_many({"metadata.document_id": document_id})
            print(f"Deleted {result.deleted_count} vectors for document {document_id}")
            return result.deleted_count
        except Exception as e:
            print(f"Error deleting vectors for document {document_id}: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            total_docs = self.collection.count_documents({})
            
            # Get user statistics
            user_stats = list(self.collection.aggregate([
                {"$group": {
                    "_id": "$metadata.user_id",
                    "count": {"$sum": 1},
                    "latest": {"$max": "$created_at"}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]))
            
            return {
                "total_documents": total_docs,
                "user_statistics": user_stats,
                "collection_name": self.collection.name,
                "is_atlas": self._is_atlas_available()
            }
            
        except Exception as e:
            print(f"Error getting vector store stats: {str(e)}")
            return {"error": str(e)}

# Singleton instance
_vector_store = None

def get_vector_store():
    """Get MongoDB vector store singleton"""
    global _vector_store
    if _vector_store is None:
        _vector_store = MongoDBVectorStore()
    return _vector_store