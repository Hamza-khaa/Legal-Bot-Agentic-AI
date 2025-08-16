from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.schema import AIMessage,HumanMessage
from langchain.agents import initialize_agent, AgentExecutor, create_react_agent
from langchain.agents.agent_types import AgentType
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any, List ,TypedDict, Annotated,Union,Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# Load and setup RAG system
loader = TextLoader("notes.txt", encoding="utf-8")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)
vectorstore = FAISS.from_documents(chunks, embeddings)

chat_history = ChatMessageHistory()

# RAG Chain setup
prompt = PromptTemplate(
    input_variables=["chat_history","context", "question"],
    template="""
You are a smart and helpful legal assistant. You are talking to a human user in a conversation.

Your goal is to:
1. Answer legal questions using the information in the context below.
2. If the context does not help, use your own general knowledge.
3. Always remember the previous conversation and facts the user shared.
4. If the user refers to something vague, use the previous conversation to figure out what they mean.
5. Be clear and helpful in your answer.

Here is the previous conversation:
{chat_history}

Context:
{context}

Question:
{question}
"""
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=True  # 🔍 VERBOSE: Track RAG chain execution
)

class IntelligentRouter:
    """Intelligent router that uses LLM to determine the appropriate agent"""
        
    def __init__(self, llm: GoogleGenerativeAI):
        self.llm = llm
        self.routing_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an intelligent router for a legal assistant system. 
            
Analyze the user's query and determine which agent should handle it:

LEGAL_RESEARCH_AGENT - Use for:
- Legal questions, definitions, law explanations
- Searching legal documents or databases
- Questions about legal concepts, procedures, rights
- Case law or statute inquiries
- Legal term explanations

DOCUMENT_DRAFTING_AGENT - Use for:
- Requests to create, write, or draft legal documents
- Making agreements, contracts, notices, letters
- Document templates or formats
- Any drafting or writing requests

GENERAL_KNOWLEDGE_AGENT - Use for:
- Non-legal questions (science, history, technology, etc.)
- General information not related to law
- Personal questions or casual conversation

Respond with ONLY the agent name: LEGAL_RESEARCH_AGENT, DOCUMENT_DRAFTING_AGENT, or GENERAL_KNOWLEDGE_AGENT"""),
            HumanMessage(content="Query: {query}")
        ])
    
    def route_query(self, query: str) -> str:
        """Route the query to the appropriate agent"""
        print(f"🔍 [ROUTER] → IntelligentRouter.route_query() processing: '{query}'")
        query_lower = query.lower()
        
        # Enhanced keyword-based routing (more reliable than LLM routing)
        
        # Document Drafting keywords
        drafting_keywords = [
            "draft", "write", "create", "make", "prepare", "compose", 
            "agreement", "contract", "notice", "letter", "document",
            "affidavit", "petition", "application", "template"
        ]
        
        # Legal Research keywords  
        legal_keywords = [
            "law", "legal", "court", "judge", "statute", "case", "section", "act",
            "explain", "what is", "define", "definition", "meaning",
            "copyright", "fraud", "theft", "criminal", "civil", "constitutional",
            "penalty", "punishment", "rights", "procedure", "jurisdiction",
            "evidence", "appeal", "bail", "custody", "arrest", "trial",
            "contract law", "tort", "liability", "damages", "injunction",
            "ipc", "crpc", "constitution", "supreme court", "high court"
        ]
        
        # Non-legal keywords
        non_legal_keywords = [
            "science", "history", "technology", "math", "physics", "chemistry",
            "biology", "geography", "weather", "sports", "movies", "music",
            "cooking", "travel", "health", "medicine", "computer", "internet"
        ]
        
        # Check for drafting
        if any(keyword in query_lower for keyword in drafting_keywords):
            print(f"🔍 [ROUTER] → Matched drafting keywords, routing to: DOCUMENT_DRAFTING_AGENT")
            return "DOCUMENT_DRAFTING_AGENT"
        
        # Check for legal research (broader check)
        if (any(keyword in query_lower for keyword in legal_keywords) or
            "explain" in query_lower or "what is" in query_lower or
            "define" in query_lower or "meaning" in query_lower):
            print(f"🔍 [ROUTER] → Matched legal keywords, routing to: LEGAL_RESEARCH_AGENT")
            return "LEGAL_RESEARCH_AGENT"
        
        # Check for clearly non-legal topics
        if any(keyword in query_lower for keyword in non_legal_keywords):
            print(f"🔍 [ROUTER] → Matched non-legal keywords, routing to: GENERAL_KNOWLEDGE_AGENT")
            return "GENERAL_KNOWLEDGE_AGENT"
        
        # Default to legal research for ambiguous queries
        print(f"🔍 [ROUTER] → No specific match, defaulting to: LEGAL_RESEARCH_AGENT")
        return "LEGAL_RESEARCH_AGENT"

# Tool functions
def query_legal_db(question: str) -> str:
    """Mock function to simulate querying a legal database"""
    if "copyright" in question.lower():
        return "Copyright penalty: Up to 3 years jail (IPC Section 420)."
    elif "theft" in question.lower():
        return "Theft penalty: Up to 3 years imprisonment (IPC Section 378-382)."
    elif "fraud" in question.lower():
        return "Fraud penalty: Up to 7 years imprisonment (IPC Section 420)."
    elif "hamza" in question.lower():
        return "Hamza is an Arabic name meaning 'strong' or 'steadfast'. In legal context, it may refer to a person's name in legal documents."
    else:
        return "No relevant law found in database. Try searching legal documents."

def rag_tool_fun(query: str, chat_history_messages: List = None) -> str:
    """RAG tool function with smart fallback - FIXED VERSION"""
    print(f"🔍 RAG Tool called with query: {query}")  # 🔍 VERBOSE: Track tool calls
    
    try:
        # Convert chat history messages to the format expected by ConversationalRetrievalChain
        if chat_history_messages is None:
            chat_history_messages = []
        
        print(f"🔍 Chat history length: {len(chat_history_messages)}")  # 🔍 VERBOSE: Track chat history
        
        # Format chat history for ConversationalRetrievalChain
        formatted_history = []
        for i in range(0, len(chat_history_messages), 2):
            if i + 1 < len(chat_history_messages):
                human_msg = chat_history_messages[i]
                ai_msg = chat_history_messages[i + 1]
                
                human_content = human_msg.content if hasattr(human_msg, 'content') else str(human_msg)
                ai_content = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
                
                formatted_history.append((human_content, ai_content))
        
        print(f"🔍 Formatted history pairs: {len(formatted_history)}")  # 🔍 VERBOSE: Track formatted history
        
        # Call qa_chain with proper chat_history format
        print("🔍 Calling qa_chain...")  # 🔍 VERBOSE: Track qa_chain call
        answer = qa_chain.invoke({
            "question": query,
            "chat_history": formatted_history  # This is the fix - providing chat_history in correct format
        })["answer"]
        
        print(f"🔍 RAG Answer length: {len(answer)} characters")  # 🔍 VERBOSE: Track answer length
        
        # Check if RAG found incomplete or no information
        if ("I don't have" in answer.lower() or 
            "The document I have access to does not mention" in answer.lower() or 
            len(answer.strip()) < 20):
            
            print("🔍 RAG answer incomplete, using fallback...")  # 🔍 VERBOSE: Track fallback usage
            # Use general knowledge as fallback for missing information
            general_answer = llm.invoke(f"Answer this legal question: {query}")
            return f"From legal documents: {answer}\n\n📚 Additional information: {general_answer}"
        
        # Check if answer seems incomplete (for mixed questions like "what is criminal law and its punishment")
        if len(answer.strip()) < 100 and any(word in query.lower() for word in ["and", "punishment", "penalty", "also"]):
            print("🔍 Answer seems incomplete for multi-part question, adding more info...")  # 🔍 VERBOSE
            # Seems like a multi-part question, get additional info
            general_answer = llm.invoke(f"Provide additional legal information about: {query}")
            return f"{answer}\n\n📚 Additional information: {general_answer}"
            
        return answer
        
    except Exception as e:
        print(f"🔍 RAG tool error: {str(e)}")  # 🔍 VERBOSE: Track errors
        # Fallback to general knowledge on error
        return llm.invoke(f"Answer this legal question: {query}")

def general_knowledge_tool_func(question: str) -> str:
    """General knowledge tool"""
    return llm.invoke(question)

def explain_legal_term(term: str) -> str:
    """Explain legal terms"""
    prompt = f"Explain the legal term '{term}' in simple words for a beginner law student. Include examples if relevant."
    return llm.invoke(prompt)

def legal_draft_tool(instruction: str) -> str:
    """Legal document drafting tool"""
    prompt = f"""Please draft a legal document based on this instruction: {instruction}. 
    
    Requirements:
    1. Use proper legal format and language
    2. Include necessary clauses and sections
    3. Add placeholder text where specific details are needed
    4. Make it professional and legally sound
    
    Instruction: {instruction}"""
    
    response = llm.invoke(prompt)
    return f"📄 Legal Draft:\n\n{response}"

# Create tools
legal_db_tool = Tool(
    name="LegalDatabase",
    func=query_legal_db,
    description="Searches legal database for specific laws and penalties. Use for specific legal violations and their punishments."
)

# Updated RAG tool to not use chat_history directly
rag_search_tool = Tool(
    name="LegalDocumentSearch",
    func=lambda query: rag_tool_fun(query, chat_history.messages),
    description="Searches uploaded legal documents and notes. Use for legal questions, definitions, and concepts from your legal knowledge base."
)

general_knowledge_tool = Tool(
    name="GeneralKnowledge",
    func=general_knowledge_tool_func,
    description="Answers general non-legal questions about science, history, technology, etc."
)

explain_legal_term_tool = Tool(
    name="ExplainLegalTerm",
    func=explain_legal_term,
    description="Explains difficult legal terms in simple language. Use when user asks about specific legal terminology.",
    return_direct=True
)

legal_draft_tool_obj = Tool(
    name="DraftLegalDocument",
    func=legal_draft_tool,
    description="Drafts legal documents like agreements, notices, contracts, affidavits based on user instructions.",
    return_direct=True
)

class LegalResearchAgent:
    """Agent specialized in legal research and explanations"""
    
    def __init__(self, llm: GoogleGenerativeAI, chat_history: ChatMessageHistory):
        self.tools = [rag_search_tool, legal_db_tool, explain_legal_term_tool]
        self.chat_history = chat_history
        
        # Create custom prompt to force tool usage
        system_prompt = """You are a legal research assistant. 

IMPORTANT RULES:
1. ALWAYS use your tools first before answering
2. For ANY legal question, FIRST search using LegalDocumentSearch tool
3. If that doesn't help, try LegalDatabase tool
4. For legal terms, use ExplainLegalTerm tool
5. Never answer legal questions without checking your tools first

You have access to these tools:
- LegalDocumentSearch: Search legal documents and notes
- LegalDatabase: Search legal database for laws and penalties  
- ExplainLegalTerm: Explain legal terms in simple language

Always try tools before giving your own answer."""

        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3,
            agent_kwargs={
                "system_message": system_prompt
            }
        )
    
    def execute(self, query: str) -> str:
        print(f"🔍 [AGENT] → LegalResearchAgent.execute() starting with: '{query}'")
        self.chat_history.add_user_message(query)
        print(f"🔍 [AGENT] → Added user message to chat history (total: {len(self.chat_history.messages)} messages)")
        
        # FIXED: Pass chat_history to the agent properly
        print(f"🔍 [AGENT] → Calling LangChain agent with conversational context...")
        result = self.agent.invoke({
            "input": query,
            "chat_history": self.chat_history.messages
        })
        
        # Extract the output from the result
        if isinstance(result, dict):
            output = result.get("output", str(result))
        else:
            output = str(result)
        
        print(f"🔍 [AGENT] → LangChain agent completed, result length: {len(output)} characters")
        self.chat_history.add_ai_message(output)
        print(f"🔍 [AGENT] → Added AI response to chat history (total: {len(self.chat_history.messages)} messages)")
        print(f"🔍 [AGENT] → LegalResearchAgent.execute() completed successfully")
        return output

class DocumentDraftingAgent:
    """Agent specialized in legal document drafting"""
    
    def __init__(self, llm: GoogleGenerativeAI, chat_history: ChatMessageHistory):
        self.tools = [legal_draft_tool_obj]
        self.chat_history = chat_history
        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=2
        )
    
    def execute(self, query: str) -> str:
        print(f"🔍 [AGENT] → DocumentDraftingAgent.execute() starting with: '{query}'")
        self.chat_history.add_user_message(query)
        print(f"🔍 [AGENT] → Added user message to chat history (total: {len(self.chat_history.messages)} messages)")
        
        # FIXED: Pass chat_history to the agent properly
        print(f"🔍 [AGENT] → Calling LangChain agent for document drafting...")
        result = self.agent.invoke({
            "input": query,
            "chat_history": self.chat_history.messages
        })
        
        # Extract the output from the result
        if isinstance(result, dict):
            output = result.get("output", str(result))
        else:
            output = str(result)
        
        print(f"🔍 [AGENT] → LangChain agent completed, result length: {len(output)} characters")
        self.chat_history.add_ai_message(output)
        print(f"🔍 [AGENT] → Added AI response to chat history (total: {len(self.chat_history.messages)} messages)")
        print(f"🔍 [AGENT] → DocumentDraftingAgent.execute() completed successfully")
        return output

class GeneralKnowledgeAgent:
    """Agent for general non-legal questions only"""
    
    def __init__(self, llm: GoogleGenerativeAI, chat_history: ChatMessageHistory):
        self.tools = [general_knowledge_tool]
        self.chat_history = chat_history
        
        system_prompt = """You are a general knowledge assistant for NON-LEGAL questions only.

You handle questions about:
- Science, technology, history, geography
- Mathematics, physics, chemistry, biology  
- General facts, trivia, explanations
- Non-legal topics

If someone asks a legal question, politely redirect them to ask about legal topics through the legal research system."""

        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=2,
            agent_kwargs={
                "system_message": system_prompt
            }
        )
    
    def execute(self, query: str) -> str:
        print(f"🔍 [AGENT] → GeneralKnowledgeAgent.execute() starting with: '{query}'")
        self.chat_history.add_user_message(query)
        print(f"🔍 [AGENT] → Added user message to chat history (total: {len(self.chat_history.messages)} messages)")
        
        # FIXED: Pass chat_history to the agent properly
        print(f"🔍 [AGENT] → Calling LangChain agent for general knowledge...")
        result = self.agent.invoke({
            "input": query,
            "chat_history": self.chat_history.messages
        })
        
        # Extract the output from the result
        if isinstance(result, dict):
            output = result.get("output", str(result))
        else:
            output = str(result)
        
        print(f"🔍 [AGENT] → LangChain agent completed, result length: {len(output)} characters")
        self.chat_history.add_ai_message(output)
        print(f"🔍 [AGENT] → Added AI response to chat history (total: {len(self.chat_history.messages)} messages)")
        print(f"🔍 [AGENT] → GeneralKnowledgeAgent.execute() completed successfully")
        return output

# **CHANGED: Updated LegalState to include route field**
class LegalState(TypedDict):
    input: str
    chat_history: Annotated[List[Union[AIMessage,HumanMessage]],add_messages]
    result: Optional[str]
    route: Optional[str]

# **CHANGED: Initialize agents globally for use in nodes**
legal_research_agent = LegalResearchAgent(llm, chat_history)
document_drafting_agent = DocumentDraftingAgent(llm, chat_history)
general_knowledge_agent = GeneralKnowledgeAgent(llm, chat_history)
intelligent_router = IntelligentRouter(llm)

# **CHANGED: Router node now uses the IntelligentRouter class**
def router_node(state: LegalState) -> LegalState:
    """Router node that determines which agent should handle the query"""
    query = state["input"]
    print(f"🔍 Router processing query: {query}")  # 🔍 VERBOSE: Track routing
    
    # Use the IntelligentRouter to determine the route
    selected_agent = intelligent_router.route_query(query)
    print(f"🔍 Router selected agent: {selected_agent}")  # 🔍 VERBOSE: Track agent selection
    
    # Map agent names to node names
    if selected_agent == "LEGAL_RESEARCH_AGENT":
        route = "legal_research_agent"
    elif selected_agent == "DOCUMENT_DRAFTING_AGENT":
        route = "document_drafting_agent"  
    elif selected_agent == "GENERAL_KNOWLEDGE_AGENT":
        route = "general_knowledge_agent"
    else:
        route = "legal_research_agent"  # Default fallback
    
    print(f"🔀 Routing to: {route}")
    return {"route": route}

# **CHANGED: New agent nodes that call the actual agents**
def legal_research_agent_node(state: LegalState) -> LegalState:
    """Legal Research Agent Node - calls the Legal Research Agent"""
    query = state["input"]
    print("📚 Legal Research Agent handling the query...")
    print(f"🔍 State chat_history length: {len(state.get('chat_history', []))}")  # 🔍 VERBOSE: Track state
    result = legal_research_agent.execute(query)
    print(f"🔍 Legal Research Agent Node completed")  # 🔍 VERBOSE: Track completion
    return {"result": result}

def document_drafting_agent_node(state: LegalState) -> LegalState:
    """Document Drafting Agent Node - calls the Document Drafting Agent"""
    query = state["input"]
    print("📝 Document Drafting Agent handling the query...")
    print(f"🔍 State chat_history length: {len(state.get('chat_history', []))}")  # 🔍 VERBOSE: Track state
    result = document_drafting_agent.execute(query)
    print(f"🔍 Document Drafting Agent Node completed")  # 🔍 VERBOSE: Track completion
    return {"result": result}

def general_knowledge_agent_node(state: LegalState) -> LegalState:
    """General Knowledge Agent Node - calls the General Knowledge Agent"""
    query = state["input"]
    print("🧠 General Knowledge Agent handling the query...")
    print(f"🔍 State chat_history length: {len(state.get('chat_history', []))}")  # 🔍 VERBOSE: Track state
    result = general_knowledge_agent.execute(query)
    print(f"🔍 General Knowledge Agent Node completed")  # 🔍 VERBOSE: Track completion
    return {"result": result}

# **CHANGED: Updated LangGraph builder with agent nodes instead of tool nodes**
builder = StateGraph(LegalState)

# Add nodes
builder.add_node("router", router_node)
builder.add_node("legal_research_agent", legal_research_agent_node)
builder.add_node("general_knowledge_agent", general_knowledge_agent_node)
builder.add_node("document_drafting_agent", document_drafting_agent_node)

# Set entry point
builder.set_entry_point("router")

# **CHANGED: Updated conditional edges to route to agent nodes**
builder.add_conditional_edges(
    "router",
    lambda state: state.get("route"),
    {
        "legal_research_agent": "legal_research_agent",
        "general_knowledge_agent": "general_knowledge_agent", 
        "document_drafting_agent": "document_drafting_agent",
    },
)

# **CHANGED: Updated edges to connect agent nodes to END**
builder.add_edge("legal_research_agent", END)
builder.add_edge("general_knowledge_agent", END)
builder.add_edge("document_drafting_agent", END)

final_langgraph_app = builder.compile()
# 👁️‍🗨️ Optional: Visualize the graph flow
final_langgraph_app.get_graph().print_ascii()

class MultiAgentLegalSystem:
    """Main orchestrator for the multi-agent legal system"""
    
    def __init__(self, llm: GoogleGenerativeAI, chat_history: ChatMessageHistory):
        self.llm = llm
        self.chat_history = chat_history
        self.router = IntelligentRouter(llm)
        
        # Initialize specialized agents
        self.legal_research_agent = LegalResearchAgent(llm, chat_history)
        self.document_drafting_agent = DocumentDraftingAgent(llm, chat_history)
        self.general_knowledge_agent = GeneralKnowledgeAgent(llm, chat_history)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query through the multi-agent system"""
        
        # Route the query
        selected_agent = self.router.route_query(query)
        
        print(f"🔀 Routing to: {selected_agent}")
        
        try:
            # Execute with appropriate agent
            if selected_agent == "LEGAL_RESEARCH_AGENT":
                result = self.legal_research_agent.execute(query)
            elif selected_agent == "DOCUMENT_DRAFTING_AGENT":
                result = self.document_drafting_agent.execute(query)
            elif selected_agent == "GENERAL_KNOWLEDGE_AGENT":
                result = self.general_knowledge_agent.execute(query)
            else:
                result = "Error: Could not route query appropriately"
            
            return {
                "query": query,
                "selected_agent": selected_agent,
                "response": result
            }
            
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            print(error_msg)
            return {
                "query": query,
                "selected_agent": selected_agent,
                "response": error_msg
            }

def main():
    """Main function to run the LangGraph-based legal bot"""
    
    print("\n🤖 Smart Legal Assistant with LangGraph!")
    print("📋 I can help with:")
    print("   • Legal research and explanations")
    print("   • Document drafting")
    print("   • General knowledge questions")
    print("\n💡 Type 'exit' to quit")
    print("🔍 VERBOSE MODE: Detailed tracking enabled\n")
    
    langgraph_chat_history = []

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break
        
        if not query.strip():
            continue
        
        print(f"\n🔍 Processing query: '{query}'")  # 🔍 VERBOSE: Track query processing
        print(f"🔍 Current chat history length: {len(langgraph_chat_history)}")  # 🔍 VERBOSE: Track history
        
        # LangGraph input state
        state = {
            "input": query,
            "chat_history": langgraph_chat_history
        }

        print("🔍 Invoking LangGraph...")  # 🔍 VERBOSE: Track LangGraph invocation
        # Run through the LangGraph
        final_state = final_langgraph_app.invoke(state)
        print("🔍 LangGraph completed")  # 🔍 VERBOSE: Track completion

        # Print and store result
        print(f"\nBot: {final_state['result']}\n")
        print("-" * 50)

        # Update chat history
        langgraph_chat_history.append(HumanMessage(content=query))
        langgraph_chat_history.append(AIMessage(content=final_state["result"]))
        print(f"🔍 Updated chat history length: {len(langgraph_chat_history)}")  # 🔍 VERBOSE: Track history update

if __name__ == "__main__":
    main()