# app/vector_store/mongodb_store.py
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from ..config import settings
from ..db.mongodb import get_database

class MongoDBVectorStore:
    """MongoDB-backed vector store for document retrieval"""
    
    def __init__(self):
        self.db = get_database()
        self.collection = self.db[settings.VECTORS_COLLECTION]
        self.embeddings = OpenAIEmbeddings()
    
    def similarity_search(self, query, k=4, user_id=None):
        """
        Search for similar documents
        
        In production with MongoDB Atlas, this would use $vectorSearch
        This is a simplified implementation for local development
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Build query filter
        query_filter = {}
        if user_id:
            query_filter["metadata.user_id"] = user_id
        
        # For demonstration - in reality we'd use MongoDB's vector search
        # Get documents (not scalable, just for demonstration)
        cursor = self.collection.find(query_filter)
        all_docs = list(cursor)
        
        # Convert back to Document objects
        documents = []
        for doc in all_docs:
            document = Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            )
            documents.append(document)
        
        # In production, you would use MongoDB Atlas Vector Search
        # For now, we're just returning k documents
        return documents[:k]

# Factory function
def get_vector_store():
    """Get MongoDB vector store singleton"""
    return MongoDBVectorStore()