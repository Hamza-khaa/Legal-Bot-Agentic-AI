#  Legal Assistant Bot — LangChain + RAG

This is an AI-powered legal assistant built with LangChain, Retrieval-Augmented Generation (RAG), and Gemini (Google Generative AI). It answers user legal questions by retrieving relevant information from legal notes and also supports drafting basic legal documents such as rent agreements, notices, and affidavits.

##  Features

-  Conversational memory with context-aware responses
-  Retrieval-based answers using FAISS vector store
-  Tools support:
  - Legal database lookup
  - RAG-based knowledge retrieval
  - Legal document drafting
  - General fallback with Gemini
-   Uses LangChain’s `ConversationalRetrievalChain` and `initialize_agent`

##  Tech Stack

- LangChain
- Google Generative AI (Gemini)
- FAISS
- Python
- dotenv
- OOP design for tool integration

##  Example Use Cases

- "What is the penalty for copyright infringement?"
- "Draft a one-year rental agreement between landlord and tenant"
- "Create an affidavit for address change"

##  How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/legal-assistant-bot.git
   cd legal-assistant-bot
2. Set up .ev
   GOOGLE_API_KEY=your_google_gemini_api_key
3. Install Dependencies
   pip install -r requirements.txt
4. Run
   python main.py

## Note:
This bot is not a substitute for legal advice.

Customize notes.txt to include your legal content.
