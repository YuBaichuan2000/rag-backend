# app/rag/engine.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from ..config import settings
from ..db.checkpointer import MongoDBCheckpointer
from ..vector_store.mongodb_store import get_vector_store

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
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=3)
            
            # Format results
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            
            return serialized, retrieved_docs
        
        # Create graph nodes
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond directly."""
            # Add system message if not present
            messages = state["messages"]
            if not any(msg.type == "system" for msg in messages):
                system_message = SystemMessage(content=
                    "You are an AI assistant that responds to questions based on stored documents. "
                    "Use the retrieval tool to find relevant information when needed. "
                    "If you don't know the answer, say so."
                )
                messages = [system_message] + messages
            
            # Bind tools to LLM
            llm_with_tools = self.llm.bind_tools([retrieve])
            
            # Get response
            response = llm_with_tools.invoke(messages)
            
            # Return updated state
            return {"messages": [response]}
        
        # Tool execution node
        tools = ToolNode([retrieve])
        
        # Response generation node
        def generate(state: MessagesState):
            """Generate answer using retrieved content."""
            # Get tool messages
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            
            # Format prompt with retrieved content
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are an AI assistant that helps users with information from their documents. "
                "Use the following retrieved information to answer the question. "
                "If you don't know the answer, say so clearly."
                "\n\n"
                f"{docs_content}"
            )
            
            # Get conversation messages (excluding tool messages)
            conversation_messages = [
                message
                for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            
            # Create prompt
            prompt = [SystemMessage(content=system_message_content)] + conversation_messages
            
            # Get response
            response = self.llm.invoke(prompt)
            
            # Return updated state
            return {"messages": [response]}
        
        # Build the graph
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node(query_or_respond)
        graph_builder.add_node(tools)
        graph_builder.add_node(generate)
        
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        
        # Set up MongoDB persistence
        mongodb_checkpointer = MongoDBCheckpointer()
        
        # Compile graph with persistence
        graph = graph_builder.compile(checkpointer=mongodb_checkpointer)
        
        return graph
    
    def process_message(self, message, thread_id=None):
        """Process a user message and return a response"""
        thread_id = thread_id or str(uuid.uuid4())
        
        # Configuration with thread ID
        config = {"configurable": {"thread_id": thread_id}}
        
        # Process the message
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        
        # Get the last AI message as the response
        ai_message = result["messages"][-1]
        response_text = ai_message.content
        
        return response_text, thread_id

# Factory function
def get_rag_engine():
    """Get RAG engine singleton"""
    return RAGEngine()