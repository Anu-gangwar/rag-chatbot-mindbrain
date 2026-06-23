import os
import requests
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
EMBED_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-albert-small-v2"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Dummy HR Policy Data - Isko apne hisaab se badal lena
HR_DOCS = [
    "Leave Policy: Employees are entitled to 24 paid leaves per year. 12 casual leaves and 12 sick leaves.",
    "Work From Home Policy: Employees can work from home 2 days per week with manager approval.",
    "Office Timings: Standard office hours are 9:30 AM to 6:30 PM, Monday to Friday.",
    "Salary Policy: Salary is credited on the last working day of every month.",
    "Notice Period: Standard notice period is 60 days for confirmed employees."
]

def get_embedding(text):
    payload = {"inputs": text}
    response = requests.post(EMBED_URL, headers=headers, json=payload)
    if response.status_code!= 200:
        return None
    return response.json()

def get_relevant_doc(query):
    # Simple keyword matching for Render free tier - no vectors needed
    query_lower = query.lower()
    for doc in HR_DOCS:
        if any(word in doc.lower() for word in query_lower.split()):
            return doc
    return HR_DOCS[0] # Default fallback

def query_llm(context, question):
    prompt = f"""<|system|>
You are MindBrain HR Policy Chatbot. Answer only from the given context. Be helpful and concise.</s>
<|user|>
Context: {context}
Question: {question}</s>
<|assistant|>
"""

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7}
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code!= 200:
        return "Sorry, I'm having trouble connecting to the AI model. Please try again."

    result = response.json()
    if isinstance(result, list):
        return result[0]['generated_text'].split("<|assistant|>")[-1].strip()
    return "Sorry, I couldn't generate a response."

@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MindBrain HR Policy Chatbot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial; max-width: 600px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
           .chat-box { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h2 { color: #333; text-align: center; }
            #chat { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
           .user { color: blue; margin: 5px 0; }
           .bot { color: green; margin: 5px 0; }
            input { width: 75%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { width: 20%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="chat-box">
            <h2>MindBrain HR Policy Chatbot</h2>
            <div id="chat"></div>
            <input type="text" id="userInput" placeholder="Ask about HR policies..." />
            <button onclick="sendMessage()">Ask</button>
        </div>

        <script>
            function sendMessage() {
                var input = document.getElementById("userInput");
                var chat = document.getElementById("chat");
                var userText = input.value;
                if (userText.trim() === "") return;

                chat.innerHTML += "<div class='user'><b>You:</b> " + userText + "</div>";
                input.value = "";
                chat.innerHTML += "<div class='bot'><b>Bot:</b> Thinking...</div>";
                chat.scrollTop = chat.scrollHeight;

                fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({message: userText})
                })
               .then(response => response.json())
               .then(data => {
                    chat.lastChild.innerHTML = "<b>Bot:</b> " + data.response;
                    chat.scrollTop = chat.scrollHeight;
                });
            }

            document.getElementById("userInput").addEventListener("keypress", function(e) {
                if (e.key === "Enter") sendMessage();
            });
        </script>
    </body>
    </html>
    """)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    context = get_relevant_doc(user_message)
    response = query_llm(context, user_message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
