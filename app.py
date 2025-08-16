# app.py - Corrected
import json
import os
import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from main import (
    IntelligentRouter,
    final_langgraph_app,
    llm
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
app.secret_key = "yoursecret"

USERS_FILE = "users.json"
user_chat_histories = {}

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password').strip()
        users = load_users()
        if username in users:
            return render_template('signup.html', error="Username already exists")
        users[username] = password
        save_users(users)
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password').strip()
        users = load_users()
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username
            if username not in user_chat_histories:
                user_chat_histories[username] = ChatMessageHistory()
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

@app.route('/chat', methods=['POST'])
def chat():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    
    username = session['username']
    user_message = request.json.get("message")
    
    if username not in user_chat_histories:
        user_chat_histories[username] = ChatMessageHistory()
        
    chat_history_for_user = user_chat_histories[username]
    
    # Run the query through the LangGraph app
    state = {
        "input": user_message,
        "chat_history": chat_history_for_user.messages
    }
    
    try:
        final_state = final_langgraph_app.invoke(state)
        response = final_state['result']
        
        # Manually update chat history after the invocation
        chat_history_for_user.add_user_message(user_message)
        chat_history_for_user.add_ai_message(response)
        
        # Determine agent used for the frontend display
        router = IntelligentRouter(llm) # We need the llm to initialize the router. This is inefficient but necessary given the constraint.
        agent_used = router.route_query(user_message)
        
        return jsonify({
            "success": True,
            "response": response,
            "agent_used": agent_used,
            "timestamp": datetime.datetime.now().strftime("%I:%M %p")
        })
        
    except Exception as e:
        print(f"Error processing chat query: {e}")
        return jsonify({
            "success": False,
            "error": "An error occurred while processing your request. Please try again."
        }), 500

@app.route('/clear', methods=['POST'])
def clear_chat():
    if not session.get('logged_in'):
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    username = session['username']
    if username in user_chat_histories:
        user_chat_histories[username].clear()
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Chat history not found."}), 404


if __name__ == "__main__":
    print("🚀 Starting Legal Bot Web Server...")
    app.run(debug=True, host="0.0.0.0", port=5000)