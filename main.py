from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")  #set up your google api in .env

# Load and split document
loader = TextLoader("notes.txt", encoding="utf-8")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embed and store
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Memory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Prompt
prompt = PromptTemplate(
    input_variables=["chat_history","context", "question"],
  template = """
You are a smart and helpful legal assistant. You are talking to a human user in a conversation.

Your goal is to:
1. Answer legal questions using the information in the context below.
2. If the context does not help, use your own general knowledge.
3. Always remember the previous conversation and facts the user shared (like their name, location, or timing).
4. If the user refers to something vague like "he", "she", "it", or "when", use the previous conversation to figure out what they mean.
5. Be clear and helpful in your answer.

Here is the previous conversation:
{chat_history}

Context:
{context}

Question:
{question}
"""
)


# LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash" , temperature=0.3)

# Chain
#  Setup Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt},

)
from langchain.tools import Tool

# Mock function to simulate querying a database
def query_legal_db(question: str) -> str:
    # In reality, this would connect to your SQL/NoSQL DB
    if "copyright" in question.lower():
        return "Copyright penalty: Up to 3 years jail (IPC Section 420)."
    else:
        return "No relevant law found in DB."

# Create the tool
legal_db_tool = Tool(
    name="LegalDatabase",
    func=query_legal_db,
    description="Searches our legal database for laws. Input: a legal question."
)

# Test the tool directly (no agent!)
user_question = "What’s the penalty for copyright infringement?"
result = legal_db_tool.run(user_question)
print(result)  # Output: "Copyright penalty: Up to 3 years jail..."


#TOOL FOR SERCHING RAG
def rag_tool_fun(query: str)->str:
    result=qa_chain.invoke({"question":query})
    return result["answer"]

rag_search_tool=Tool(
    name="LegalSearchTool",
    func=rag_tool_fun,
     description=(
        "Use this to answer ANY legal questions, definitions, laws, or terms "
        "based on the uploaded legal notes and documents. "
        "Use this before trying anything else for legal topics."
    )
)
def general_knowledge_tool_func(question: str) -> str:
    return llm.invoke(question)

general_knowledge_tool = Tool(
    name="QuestionsOtherthanLaw",
    func=general_knowledge_tool_func,
     description="Always use this tool to answer general (non-legal) questions such as science, history, technology, etc."
)


def explain_legal_term(term: str)->str:
    prompt=f"Explain the legal term {term} for a beginer law student"
    return llm.invoke(prompt)

explain_legal_term_tool=Tool(
    name="ExplainTerm",
    func=explain_legal_term,
    description="Explains difficult legal terms in simple words. Input: a legal term like 'res judicata', 'mens rea'.",
    return_direct=True
)

def legal_draft_tool(instruction: str)->str:
    prompt=f"Please draft a legal document based on this instruction: {instruction}. Keep it in proper legal format."
    response=llm.invoke(prompt)
    return f" Legal Draft:\n\n{response}"
legal_draft_tool=Tool(
    name="DraftDocument",
    func=legal_draft_tool,
    description="Drafts basic legal documents like rent agreements, notices, affidavits. Input: drafting instructions.",
    return_direct=True
)

agent_executor= initialize_agent(
    tools = [rag_search_tool,legal_db_tool ,explain_legal_term_tool,general_knowledge_tool,legal_draft_tool],
    # tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    


)

print("\n Ask your legal assistant! (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    # Let the agent decide h
    result = agent_executor.invoke({"input": query})
    print(result["output"])  # This includes the final agent output


