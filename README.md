# Smart-Legal-AI-Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-0.1.16-blueviolet?style=for-the-badge" alt="LangChain">
  <img src="https://img.shields.io/badge/LangGraph-0.0.30-orange?style=for-the-badge" alt="LangGraph">
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/Google_Gemini-2.0-F9A52A?style=for-the-badge&logo=google&logoColor=white" alt="Google Gemini">
</p>

## 🌟 Project Overview

**Smart-Legal-AI-Assistant** is a powerful, web-based AI assistant designed to simplify legal research and document drafting. Built on a sophisticated multi-agent system, it intelligently routes user queries to specialized AI agents, ensuring accurate, relevant, and efficient responses. The system uses a LangGraph-powered orchestrator to manage the flow, providing a seamless and dynamic conversational experience.

## ✨ Key Features

- **Multi-Agent Architecture**: The core of the application is a robust multi-agent system. An intelligent router analyzes the user's input and directs it to the most appropriate specialized agent.
- **Specialized AI Agents**:
    - 📚 **Legal Research Agent**: Designed for legal questions. It uses a Retrieval-Augmented Generation (RAG) system to search an internal knowledge base (`notes.txt`) and a mock legal database for information on laws, penalties, and legal concepts.
    - 📝 **Document Drafting Agent**: Specializes in creating legal documents, such as agreements, contracts, and notices, based on user instructions.
    - 🧠 **General Knowledge Agent**: A fallback agent that handles non-legal or casual conversation, ensuring the system can engage with a wide range of topics.
- **Retrieval-Augmented Generation (RAG)**: The Legal Research Agent leverages LangChain, Google Gemini Embeddings, and a FAISS vector store to perform semantic searches on a custom legal knowledge base, providing context-aware and accurate answers.
- **LangGraph-Powered Orchestration**: The entire workflow—from routing to agent execution—is orchestrated using LangGraph, providing a clear, stateful, and visible flow for complex conversational tasks.
- **Conversational Memory**: The system maintains chat history for each user, allowing for a continuous and coherent conversation where agents can remember and reference previous interactions.
- **Interactive Web Interface**: A clean and modern frontend built with Flask, HTML, CSS, and JavaScript provides a user-friendly chat interface with real-time feedback, a welcome message with feature highlights, and clear `user` and `bot` message bubbles.
- **User Authentication**: A basic user authentication system is implemented with `login.html` and `signup.html`, storing user data in a `users.json` file. This allows for personalized chat history for each user.
- **Scalable Technology Stack**: Built with Python, Flask, LangChain, and LangGraph, the project uses Google's powerful Gemini 2.0 Flash model and is structured for future expansion and integration with more advanced legal tools.

## 📁 Project Structure

├── pycache/           
├── templates/
│   ├── index.html         
│   ├── login.html         
│   └── signup.html      
├── venv/                 
├── .env                   
├── app.py                 
├── main.py                
├── notes.txt              
├── requirements.txt       
└── users.json             

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd Smart-Legal-AI-Assistant
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
    -   **On Windows:**
        ```bash
        venv\Scripts\activate
        ```
    -   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` file should include `langchain`, `langgraph`, `flask`, `python-dotenv`, `langchain-google-genai`, `faiss-cpu`, and `tiktoken`.*

4.  **Set up your API Key:**
    -   Create a `.env` file in the root directory of the project.
    -   Add your Google API Key:
    ```
    GOOGLE_API_KEY="your-google-api-key"
    ```
    -   You can obtain a key from the [Google AI Studio](https://aistudio.google.com/app/apikey).

5.  **Prepare the legal knowledge base:**
    -   The RAG system uses a `notes.txt` file. Ensure this file is present and contains the legal information you want the assistant to reference.

6.  **Create user data file:**
    -   The login system expects a `users.json` file. Create an empty JSON object `{}` in this file for new users to sign up.

## 🚀 How to Run

1.  **Start the Flask web server:**
    ```bash
    python app.py
    ```
2.  **Access the application:**
    -   Open your web browser and navigate to `http://127.0.0.1:5000` or `http://localhost:5000`.
3.  **Use the assistant:**
    -   Sign up for a new account.
    -   Log in and start a conversation.
    -   Try asking questions like:
        -   `"What is a non-disclosure agreement?"` (Legal Research)
        -   `"Draft a letter of intent to purchase property."` (Document Drafting)
        -   `"What is the capital of France?"` (General Knowledge)
        -   `"Explain the difference between civil and criminal law."` (Legal Research)

## 🤝 Contribution

Contributions are welcome! If you have suggestions for new features, improvements, or bug fixes, feel free to open an issue or submit a pull request.
