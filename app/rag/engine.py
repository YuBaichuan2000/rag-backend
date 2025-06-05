# app/rag/engine.py - Updated to use MongoDB Vector Store
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from ..config import settings
from ..vector_store import get_vector_store  # ✅ Use factory pattern
from ..db.mongodb import get_database
from typing import Optional, List

class RAGEngine:
    """RAG Engine using LangGraph with MongoDB Vector Store"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=settings.LLM_TEMPERATURE, 
            model=settings.LLM_MODEL
        )
        
        # Use the factory pattern to get the correct vector store
        print(f"🔗 Initializing RAG Engine with {settings.VECTOR_STORE_TYPE} vector store...")
        self.vector_store = get_vector_store()
        print(f"✅ Vector store initialized: {type(self.vector_store).__name__}")
        
        # Log vector store statistics if available
        if hasattr(self.vector_store, 'get_stats'):
            try:
                stats = self.vector_store.get_stats()
                print(f"📊 Vector store contains {stats.get('total_documents', 0)} documents")
            except Exception as e:
                print(f"⚠️ Could not get vector store stats: {e}")
        
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph for RAG with proper tool handling"""
        # Create retrieval tool with enhanced user support
        @tool()
        def retrieve(query: str, user_id: Optional[str] = None):
            """Retrieve information related to a query for a specific user."""
            try:
                print(f"🔍 Retrieve tool called with query: '{query}' for user: {user_id}")
                
                # Use user_id if provided, otherwise search all documents
                retrieved_docs = self.vector_store.similarity_search(
                    query, 
                    k=settings.SIMILARITY_SEARCH_K, 
                    user_id=user_id
                )
                
                if not retrieved_docs:
                    print("❌ No documents found for query")
                    if user_id:
                        return f"No relevant documents found for this query in your personal knowledge base. You may want to upload some documents first."
                    else:
                        return "No relevant documents found for this query in the knowledge base."
                
                print(f"✅ Retrieved {len(retrieved_docs)} documents")
                
                # Format results with better structure
                formatted_results = []
                for i, doc in enumerate(retrieved_docs, 1):
                    # Get similarity score if available
                    score = doc.metadata.get('similarity_score', 'N/A')
                    source = doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
                    doc_id = doc.metadata.get('document_id', 'Unknown')
                    
                    result = f"Document {i}:\n"
                    result += f"Source: {source}\n"
                    result += f"Similarity: {score}\n"
                    result += f"Content: {doc.page_content}\n"
                    
                    formatted_results.append(result)
                
                serialized = "\n" + "="*50 + "\n".join(formatted_results)
                
                print(f"📄 Sample content preview: {retrieved_docs[0].page_content[:100]}...")
                return serialized
                
            except Exception as e:
                print(f"❌ Error in retrieve tool: {str(e)}")
                import traceback
                print(f"📋 Traceback: {traceback.format_exc()}")
                return f"Error retrieving documents: {str(e)}"
        
        # Choose checkpointer based on configuration
        if settings.VECTOR_STORE_TYPE == "mongodb":
            try:
                # Try to use MongoDB checkpointer for consistency
                print("🔄 Attempting to initialize MongoDB checkpointer...")
                
                # Get MongoDB connection details
                db = get_database()
                
                # Create MongoDB checkpointer
                # Note: MongoDBSaver requires specific setup
                checkpointer = MongoDBSaver(
                    client=db,
                    collection_name="langgraph_checkpoints"
                )

                print("✅ Using MongoDB checkpointer for LangGraph state persistence")
                
            except Exception as e:
                print(f"⚠️ Could not initialize MongoDB checkpointer: {e}")
                print("🔄 Falling back to InMemory checkpointer")
                checkpointer = InMemorySaver()
        else:
            # Use memory-based checkpointer for FAISS or other stores
            checkpointer = InMemorySaver()
            print(f"💾 Using InMemory checkpointer")
        
        # Create the LLM with tools
        llm_with_tools = self.llm.bind_tools([retrieve])
        
        # Define the call model function with user context
        def call_model(state):
            """Process messages and generate a response"""
            messages = state["messages"]
            print(f"🧠 call_model received {len(messages)} messages")
            
            # Debug: Log message types in current state
            for i, msg in enumerate(messages[-3:]):  # Show last 3 messages
                content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
                print(f"  📝 Message {i}: {msg.type.upper()} - {content_preview}")
            
            # Enhanced system message with more context
            if not any(msg.type == "system" for msg in messages):
                system_message = SystemMessage(content=
                    "You are a helpful AI assistant with access to a knowledge base through document retrieval. "
                    f"You are using a {settings.VECTOR_STORE_TYPE} vector store for document search. "
                    "When users ask questions, use the retrieve tool to find relevant information from the stored documents. "
                    "If you find relevant documents, base your answer on that information and cite the sources. "
                    "If no relevant documents are found, let the user know no information is found in the database."
                    "Always be helpful, accurate, and cite your sources when using retrieved information."
                )
                messages = [system_message] + messages
            
            # Generate response
            response = llm_with_tools.invoke(messages)
            
            # Return updated state (MessagesState automatically appends)
            return {"messages": [response]}
        
        # Tool execution node
        tools_node = ToolNode(tools=[retrieve])
        
        # Build the graph
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", tools_node)
        
        # Set entry point
        builder.set_entry_point("call_model")
        
        # Add conditional edges
        builder.add_conditional_edges(
            "call_model",
            tools_condition,
            {
                "tools": "tools",  # If tool calls are present, go to tools
                END: END,          # Otherwise, end
            },
        )
        
        # Add edge from tools back to model
        builder.add_edge("tools", "call_model")
        
        # Compile graph with persistence
        graph = builder.compile(checkpointer=checkpointer)
        
        print("✅ LangGraph RAG engine compiled successfully")
        return graph
    
    def process_message(self, message: str, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> tuple[str, str]:
        """
        Process message using LangGraph with user context
        Enhanced to support user-specific document retrieval
        """
        try:
            thread_id = thread_id or str(uuid.uuid4())
            print(f"\n🆕 === LANGGRAPH RAG PROCESSING ===")
            print(f"🔗 Thread ID: {thread_id}")
            print(f"👤 User ID: {user_id}")
            print(f"🏪 Vector Store: {settings.VECTOR_STORE_TYPE}")
            print(f"📝 Message: {message[:100]}...")
            
            # Configuration with thread_id for LangGraph persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            # Add user context to the message if provided
            if user_id:
                # We can pass user context through the message or state
                # For now, we'll add it as metadata in the human message
                human_message = HumanMessage(
                    content=message,
                    additional_kwargs={"user_id": user_id}
                )
            else:
                human_message = HumanMessage(content=message)
            
            input_state = {"messages": [human_message]}
            print(f"📤 Sending to LangGraph with user context")
            
            # Process the message using LangGraph's state management
            print(f"🚀 Invoking LangGraph RAG engine...")
            result = self.graph.invoke(input_state, config=config)
            print(f"✅ LangGraph processing completed")
            print(f"📥 Result contains {len(result['messages'])} total messages")
            
            # Debug: Show the conversation flow
            print(f"🔍 === CONVERSATION FLOW ===")
            for i, msg in enumerate(result["messages"][-5:]):  # Show last 5 messages
                content_preview = str(msg.content)[:80] + "..." if len(str(msg.content)) > 80 else str(msg.content)
                tool_info = ""
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_info = f" [🛠️ {len(msg.tool_calls)} tool calls]"
                elif hasattr(msg, "tool_call_id"):
                    tool_info = f" [🔧 tool response]"
                print(f"  {i+1}: {msg.type.upper()}{tool_info} - {content_preview}")
            
            # Get the last AI message as the response
            ai_message = result["messages"][-1]
            response_text = ai_message.content
            
            print(f"💬 Final response length: {len(response_text)} characters")
            print(f"📄 Response preview: {response_text[:150]}...")
            print(f"✅ === RAG PROCESSING COMPLETED ===\n")
            
            return response_text, thread_id
            
        except Exception as e:
            print(f"❌ Error in RAG processing: {str(e)}")
            import traceback
            print(f"📋 Full traceback:")
            print(traceback.format_exc())
            
            # Provide a helpful fallback response
            error_response = (
                f"I encountered a technical issue while processing your request: {str(e)}. "
                f"This might be due to vector store connectivity or configuration issues. "
                f"Please check that your {settings.VECTOR_STORE_TYPE} vector store is properly configured."
            )
            
            return error_response, thread_id

    def get_vector_store_info(self) -> dict:
        """Get information about the current vector store"""
        try:
            info = {
                "type": settings.VECTOR_STORE_TYPE,
                "class": type(self.vector_store).__name__
            }
            
            if hasattr(self.vector_store, 'get_stats'):
                stats = self.vector_store.get_stats()
                info.update(stats)
            
            return info
        except Exception as e:
            return {"type": settings.VECTOR_STORE_TYPE, "error": str(e)}

# Singleton pattern
_rag_engine = None

def get_rag_engine():
    """Get RAG engine singleton"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine

# Utility function to reinitialize the engine (useful after vector store changes)
def reinitialize_rag_engine():
    """Reinitialize the RAG engine (useful after configuration changes)"""
    global _rag_engine
    _rag_engine = None
    return get_rag_engine()