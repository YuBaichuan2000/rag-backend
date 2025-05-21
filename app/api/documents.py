# app/api/documents.py
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Body
from typing import Optional

from ..models.api_models import DocumentUploadResponse, URLUploadRequest
from ..document_processing.loaders import (
    load_document_from_url,
    load_document_from_pdf,
    load_document_from_text
)
from ..document_processing.processor import process_and_store_documents

router = APIRouter()

@router.post("/upload-url", response_model=DocumentUploadResponse)
async def upload_url(request: URLUploadRequest):
    """Upload a document from a URL with enhanced debugging"""
    print(f"\n==== URL UPLOAD ATTEMPT ====")
    print(f"URL: {request.url}")
    print(f"User ID: {request.user_id}")
    print(f"Title: {request.title or 'Not provided'}")
    
    try:
        print(f"Step 1: Loading document from URL")
        documents = await load_document_from_url(request.url, request.title)
        print(f"✅ Successfully loaded document from URL")
        print(f"Document count: {len(documents)}")
        if documents:
            print(f"First document content length: {len(documents[0].page_content)}")
            print(f"First document metadata: {documents[0].metadata}")
        
        print(f"Step 2: Processing and storing documents")
        document_ids = await process_and_store_documents(documents, request.user_id)
        print(f"✅ Successfully processed and stored documents")
        print(f"Document IDs: {document_ids}")
        
        # Return information about the stored document
        return DocumentUploadResponse(
            document_id=document_ids[0],
            title=documents[0].metadata.get("title", "Unknown"),
            status="success",
            message=f"Successfully uploaded and processed document from URL"
        )
    
    except Exception as e:
        import traceback
        print(f"❌ ERROR in upload_url: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@router.post("/upload-file", response_model=DocumentUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    user_id: str = Form(...)
):
    """Upload a document file (PDF or text)"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process based on file type
        if file.filename.lower().endswith('.pdf'):
            documents = await load_document_from_pdf(file_content, file.filename, title)
        elif file.filename.lower().endswith(('.txt', '.md')):
            # Convert bytes to string for text files
            text_content = file_content.decode('utf-8')
            documents = await load_document_from_text(text_content, file.filename, title)
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload PDF or text files."
            )
        
        # Process and store documents
        document_ids = await process_and_store_documents(documents, user_id)
        
        # Return information about the stored document
        return DocumentUploadResponse(
            document_id=document_ids[0],
            title=documents[0].metadata.get("title", "Unknown"),
            status="success",
            message=f"Successfully uploaded and processed {file.filename}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")