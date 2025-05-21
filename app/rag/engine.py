# app/rag/engine.py
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
# from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from ..config import settings
from ..vector_store.faiss_store import get_vector_store

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
        """Build the LangGraph for RAG"""
        # Create retrieval tool
        @tool()
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=3)
            
            # Format results
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            
            return serialized
        
        # Create MongoDB checkpointer
        # Use LangGraph's MongoDB checkpointer
        # mongodb_checkpointer = MongoDBCheckpointer(
        #     connection_string=settings.MONGODB_CONNECTION_STRING,
        #     database_name=settings.DB_NAME,
        #     collection_name=settings.CHAT_HISTORY_COLLECTION
        # )

       
        
        # Create the LLM with tools
        llm_with_tools = self.llm.bind_tools([retrieve])
        
        # Define the model call function
        def call_model(state: MessagesState):
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
        
        # Build the graph
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "call_model")
        
        # Compile graph with persistence
        # mongodb_checkpointer = MongoDBSaver.from_conn_string(settings.MONGODB_CONNECTION_STRING)
        checkpointer = InMemorySaver()
        
        graph = builder.compile(checkpointer=checkpointer)
        
        return graph
    
    def process_message(self, message, thread_id=None):
        """Process a user message and return a response"""
        try:
            thread_id = thread_id or str(uuid.uuid4())
            print(f"Processing message with thread_id: {thread_id}")
            
            # Configuration with thread_id
            config = {"configurable": {"thread_id": thread_id}}
            print(f"LangGraph config: {config}")
            
            # Process the message
            print(f"Invoking LangGraph with message: {message[:50]}...")
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config
            )
            print(f"LangGraph result obtained: {result.keys()}")
            
            # Get the last AI message as the response
            ai_message = result["messages"][-1]
            response_text = ai_message.content
            
            return response_text, thread_id
        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

# Factory function
def get_rag_engine():
    """Get RAG engine singleton"""
    return RAGEngine()