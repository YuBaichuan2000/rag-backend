# app/rag/engine.py
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from ..config import settings
from ..vector_store.faiss_store import get_vector_store
from ..db.mongodb import get_database
from typing import Optional, List

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
        print(f"💾 Using InMemorySaver checkpointer")

        # checkpointer = MongoDBSaver()
        
        # Create the LLM with tools
        llm_with_tools = self.llm.bind_tools([retrieve])
        
        # Define the call model function 
        def call_model(state):
            """Process messages and generate a response"""
            messages = state["messages"]
            print(f"🧠 call_model received {len(messages)} messages")
            # Debug: Log message types in current state
            for i, msg in enumerate(messages[-3:]):  # Show last 3 messages
                print(f"  📝 Message {i}: {msg.type} - {str(msg.content)[:50]}...")
            
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
            
            # Return updated state (MessagesState automatically appends)
            return {"messages": [response]}
        
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
    
    def process_message(self, message: str, thread_id: Optional[str] = None) -> tuple[str, str]:
        """
        NEW: Process message using LangGraph's native message management
        This is the preferred method that leverages LangGraph's built-in state handling
        """
        try:
            thread_id = thread_id or str(uuid.uuid4())
            print(f"\n🆕 === LANGGRAPH NATIVE METHOD ===")
            print(f"🔗 Processing message with thread_id: {thread_id}")
            print(f"📝 Message: {message[:100]}...")
            
            # Configuration with thread_id for LangGraph persistence
            config = {"configurable": {"thread_id": thread_id}}
            print(f"⚙️ LangGraph config: {config}")
            
            # Let LangGraph handle the message history automatically
            # We only pass the new human message - LangGraph retrieves existing history
            input_state = {"messages": [HumanMessage(content=message)]}
            print(f"📤 Sending to LangGraph: 1 new HumanMessage")
            
            # Process the message using LangGraph's state management
            print(f"🚀 Invoking LangGraph...")
            result = self.graph.invoke(input_state, config=config)
            print(f"✅ LangGraph completed successfully")
            print(f"📥 Result contains {len(result['messages'])} total messages")
            
            # Debug: Show the conversation flow
            print(f"🔍 === CONVERSATION FLOW DEBUG ===")
            for i, msg in enumerate(result["messages"][-5:]):  # Show last 5 messages
                msg_preview = str(msg.content)[:80] + "..." if len(str(msg.content)) > 80 else str(msg.content)
                tool_info = ""
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_info = f" [🛠️ {len(msg.tool_calls)} tool calls]"
                elif hasattr(msg, "tool_call_id"):
                    tool_info = f" [🔧 tool response to {msg.tool_call_id[:8]}...]"
                print(f"  {i}: {msg.type.upper()}{tool_info} - {msg_preview}")
            
            # Get the last AI message as the response
            ai_message = result["messages"][-1]
            response_text = ai_message.content
            
            print(f"💬 Final response: {response_text[:100]}...")
            print(f"✅ === LANGGRAPH NATIVE METHOD COMPLETED ===\n")
            
            return response_text, thread_id
            
        except Exception as e:
            print(f"❌ Error in LangGraph native method: {str(e)}")
            import traceback
            print(f"📋 Traceback: {traceback.format_exc()}")
            
            # Provide a fallback response
            return f"I encountered a technical issue with the new system: {str(e)}", thread_id

# Factory function
def get_rag_engine():
    """Get RAG engine singleton"""
    return RAGEngine()