# app/document_processing/loaders.py
import os
import tempfile
from datetime import datetime
from typing import Optional, List
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader

async def load_document_from_url(url: str, title: Optional[str] = None) -> List[Document]:
    """Load document from URL with enhanced debugging"""
    print(f"load_document_from_url: Starting with URL {url}")
    try:
        print(f"Creating WebBaseLoader for URL: {url}")
        loader = WebBaseLoader(url)
        print(f"Calling loader.load()")
        documents = loader.load()
        print(f"✅ Successfully loaded {len(documents)} documents from URL")
        
        # Set title if provided or extract from URL
        doc_title = title or url.split("/")[-1]
        print(f"Setting document title: {doc_title}")
        
        # Update metadata
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source": url,
                "title": doc_title,
                "type": "url",
                "date_added": datetime.now()
            })
            print(f"Updated metadata for document {i+1}")
        
        return documents
    except Exception as e:
        import traceback
        print(f"❌ ERROR in load_document_from_url: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Failed to load URL: {str(e)}")

async def load_document_from_pdf(file_content: bytes, filename: str, title: Optional[str] = None) -> List[Document]:
    """Load document from PDF file"""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Set title if provided or use filename
        doc_title = title or filename
        
        # Update metadata
        for doc in documents:
            doc.metadata.update({
                "source": filename,
                "title": doc_title,
                "type": "pdf",
                "date_added": datetime.now()
            })
        
        return documents
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Failed to load PDF: {str(e)}")

async def load_document_from_text(file_content: str, filename: str, title: Optional[str] = None) -> List[Document]:
    """Load document from text file"""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(file_content.encode('utf-8'))
            tmp_path = tmp_file.name
        
        # Load text file
        loader = TextLoader(tmp_path)
        documents = loader.load()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Set title if provided or use filename
        doc_title = title or filename
        
        # Update metadata
        for doc in documents:
            doc.metadata.update({
                "source": filename,
                "title": doc_title,
                "type": "text",
                "date_added": datetime.now()
            })
        
        return documents
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Failed to load text file: {str(e)}")