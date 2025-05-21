# app/vector_store/faiss_store.py
import os
import pickle
import uuid
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..config import settings

class FAISSVectorStore:
    """FAISS-backed vector store for document retrieval"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self._vector_store = None
        self._initialize_store()
        print("Initialized FAISS vector store")
        
    def _initialize_store(self):
        """Initialize an empty FAISS vector store"""
        # Create an empty FAISS index
        self._vector_store = FAISS.from_documents(
            [Document(page_content="", metadata={})],  # Dummy document
            self.embeddings
        )
        # Reset the index after initialization
        self._vector_store.index = None
        self._vector_store.docstore = {}
        
    def add_documents(self, documents: List[Document], user_id: Optional[str] = None):
        """Add documents to the FAISS vector store"""
        if not documents:
            print("No documents to add")
            return
            
        print(f"Adding {len(documents)} documents to FAISS store")
        
        # Add user_id to metadata if provided
        if user_id:
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["user_id"] = user_id
        
        # If this is the first addition, create the store
        if self._vector_store.index is None or len(self._vector_store.docstore) == 0:
            self._vector_store = FAISS.from_documents(documents, self.embeddings)
            print("Created new FAISS index")
        else:
            # Add documents to existing store
            print("Adding to existing FAISS index")
            self._vector_store.add_documents(documents)
        
        print(f"FAISS index now contains {len(self._vector_store.docstore)} documents")
    
    def similarity_search(self, query: str, k: int = 4, user_id: Optional[str] = None):
        """Search for similar documents"""
        print(f"Searching FAISS for: '{query}' (k={k}, user_id={user_id})")
        
        # If the index is empty, return empty results
        if self._vector_store.index is None or len(self._vector_store.docstore) == 0:
            print("FAISS index is empty, returning no results")
            return []
        
        # Perform search
        if user_id:
            # Filter by user_id
            filter_dict = {"user_id": user_id}
            try:
                docs = self._vector_store.similarity_search(
                    query, k=k, filter=filter_dict
                )
            except Exception as e:
                print(f"Error in FAISS search with filter: {str(e)}")
                # Fall back to unfiltered search
                docs = self._vector_store.similarity_search(query, k=k)
                # Manually filter results
                docs = [doc for doc in docs if doc.metadata.get("user_id") == user_id][:k]
        else:
            # No filter
            docs = self._vector_store.similarity_search(query, k=k)
        
        print(f"Found {len(docs)} similar documents")
        return docs
    
    def save_local(self, folder_path: str = "faiss_index"):
        """Save the FAISS index locally"""
        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Save the index
        self._vector_store.save_local(folder_path)
        print(f"FAISS index saved to {folder_path}")
    
    def load_local(self, folder_path: str = "faiss_index"):
        """Load the FAISS index from disk"""
        try:
            self._vector_store = FAISS.load_local(folder_path, self.embeddings)
            print(f"FAISS index loaded from {folder_path}")
            print(f"Index contains {len(self._vector_store.docstore)} documents")
        except Exception as e:
            print(f"Error loading FAISS index: {str(e)}")
            # Initialize a new one
            self._initialize_store()

# Singleton instance
_vector_store = None

# Factory function
def get_vector_store():
    """Get FAISS vector store singleton"""
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
    return _vector_store