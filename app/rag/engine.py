# app/rag/engine.py
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
# from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from ..config import settings
from ..vector_store.faiss_store import get_vector_store
from ..db.mongodb import get_database

# Dictionary to store thread states
THREAD_MESSAGES = {}

class RAGEngine:
    """RAG Engine using LangGraph"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=settings.LLM_TEMPERATURE, 
            model=settings.LLM_MODEL
        )
        self.vector_store = get_vector_store()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph for RAG with proper tool handling"""
        # Create retrieval tool
        @tool()
        def retrieve(query: str):
            """Retrieve information related to a query."""
            try:
                print(f"Retrieve tool called with query: '{query}'")
                retrieved_docs = self.vector_store.similarity_search(query, k=3)
                
                if not retrieved_docs:
                    print("No documents found for query")
                    return "No relevant documents found for this query."
                    
                # Format results
                serialized = "\n\n".join(
                    (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                    for doc in retrieved_docs
                )
                
                print(f"Retrieved {len(retrieved_docs)} documents")
                return serialized
            except Exception as e:
                print(f"Error in retrieve tool: {str(e)}")
                return f"Error retrieving documents: {str(e)}"
        
        # Use memory-based checkpointer for simplicity
        checkpointer = InMemorySaver()
        
        # Create the LLM with tools
        llm_with_tools = self.llm.bind_tools([retrieve])
        
        # Define the call model function 
        def call_model(state):
            """Process messages and generate a response"""
            messages = state["messages"]
            
            # Add system message if not present
            if not any(msg.type == "system" for msg in messages):
                system_message = SystemMessage(content=
                    "You are an AI assistant that responds to questions based on stored documents. "
                    "Use the retrieval tool to find relevant information when needed. "
                    "If you don't know the answer, say so."
                )
                messages = [system_message] + messages
            
            # Generate response
            response = llm_with_tools.invoke(messages)
            
            # Return updated state
            return {"messages": messages + [response]}
        
        # Tool execution node
        tools_node = ToolNode(tools=[retrieve])
        
        
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
        
        return graph
    
    def process_message(self, message, thread_id=None):
        """Process a user message and return a response with tool call handling"""
        try:
            thread_id = thread_id or str(uuid.uuid4())
            print(f"Processing message with thread_id: {thread_id}")
            
            # Get or initialize message history for this thread
            if thread_id in THREAD_MESSAGES:
                messages = THREAD_MESSAGES[thread_id].copy()  # Create a copy to work with
                
                # CRITICAL FIX: Check if the last message is an assistant message with tool calls but no tool responses
                if len(messages) > 0 and messages[-1].type == "ai" and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
                    print("Detected incomplete tool call sequence, cleaning up...")
                    # Remove the last message with incomplete tool calls
                    messages.pop()
                    print(f"Removed incomplete message, now have {len(messages)} messages")
                
                # Add the new message
                messages.append(HumanMessage(content=message))
                print(f"Added message to existing thread with {len(messages)} messages")
            else:
                # Start a new conversation with system message
                system_message = SystemMessage(content=
                    "You are an AI assistant that responds to questions based on stored documents. "
                    "Use the retrieval tool to find relevant information when needed. "
                    "ALWAYS use the retrieval tool first when answering questions about documents or topics "
                    "that might be in the user's documents. If you don't know the answer, say so."
                )
                messages = [system_message, HumanMessage(content=message)]
                print(f"Created new thread with system message")
            
            # Configuration with thread_id
            config = {"configurable": {"thread_id": thread_id}}
            print(f"LangGraph config: {config}")
            
            # Process the message
            print(f"Invoking LangGraph with message: {message[:50]}...")
            try:
                result = self.graph.invoke(
                    {"messages": messages},
                    config=config
                )
                print(f"LangGraph result obtained: {result.keys()}")
                
                # Debug: Check the last few messages to ensure tool calls are properly paired
                last_messages = result["messages"][-3:] if len(result["messages"]) >= 3 else result["messages"]
                for i, msg in enumerate(last_messages):
                    print(f"Message {i}: type={msg.type}")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"  Has tool_calls: {len(msg.tool_calls)}")
                        for tc in msg.tool_calls:
                            print(f"  Tool call: {tc.get('id', 'unknown')}")
                
                # Store the updated messages
                THREAD_MESSAGES[thread_id] = result["messages"]
                print(f"Stored updated thread with {len(THREAD_MESSAGES[thread_id])} messages")
                
                # Get the last AI message as the response
                ai_message = result["messages"][-1]
                response_text = ai_message.content
                
                return response_text, thread_id
                
            except Exception as e:
                print(f"Error during LangGraph invoke: {str(e)}")
                error_message = str(e).lower()
                
                # Handle tool call errors
                if "tool_call" in error_message and "must be followed by tool messages" in error_message:
                    print("TOOL CALL ERROR: Cleaning up conversation history...")
                    
                    # Find and remove problematic messages
                    clean_messages = []
                    pending_tool_calls = set()
                    
                    for msg in messages:
                        # Track tool calls from AI
                        if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                if "id" in tc:
                                    pending_tool_calls.add(tc["id"])
                        
                        # Track tool responses
                        if msg.type == "tool" and hasattr(msg, "tool_call_id"):
                            if msg.tool_call_id in pending_tool_calls:
                                pending_tool_calls.remove(msg.tool_call_id)
                        
                        # Keep this message if it's not an AI message with pending tool calls
                        if not (msg.type == "ai" and hasattr(msg, "tool_calls") and 
                            any(tc.get("id") in pending_tool_calls for tc in msg.tool_calls)):
                            clean_messages.append(msg)
                    
                    # Try again with cleaned messages
                    messages = clean_messages + [HumanMessage(content=message)]
                    print(f"Retrying with cleaned conversation ({len(messages)} messages)...")
                    result = self.graph.invoke(
                        {"messages": messages},
                        config=config
                    )
                    
                    # Store the updated messages
                    THREAD_MESSAGES[thread_id] = result["messages"]
                    print(f"Stored updated thread with {len(THREAD_MESSAGES[thread_id])} messages")
                    
                    # Get the last AI message as the response
                    ai_message = result["messages"][-1]
                    response_text = ai_message.content
                    
                    return response_text, thread_id
                
                # For other errors, just raise
                raise
                
        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # Provide a fallback response
            return "I encountered a technical issue. Please try again or rephrase your question.", thread_id

# Factory function
def get_rag_engine():
    """Get RAG engine singleton"""
    return RAGEngine()