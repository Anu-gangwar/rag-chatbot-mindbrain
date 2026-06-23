import os
import requests
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

HR_DOCS = [
    "Leave Policy: Employees are entitled to 24 paid leaves per year. 12 casual leaves and 12 sick leaves.",
    "Work From Home Policy: Employees can work from home 2 days per week with manager approval.",
    "Office Timings: Standard office hours are 9:30 AM to 6:30 PM, Monday to Friday.",
    "Salary Policy: Salary is credited on the last working day of every month.",
    "Notice Period: Standard notice period is 60 days for confirmed employees."
]

def get_relevant_doc(query):
    query_lower = query.lower()
    for doc in HR_DOCS:
        if any(word in doc.lower() for word in query_lower.split()):
            return doc
    return "General HR Policy: For specific queries please contact HR department."

def query_gemini(context, question):
    prompt = f"""You are MindBrain HR Policy Chatbot. Answer ONLY from the context given below. Be helpful and concise.

Context: {context}

Question: {question}
Answer:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 200}
    }

    try:
        response = requests.post(GEMINI_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return f"API error: {response.status_code}. Check GEMINI_API_KEY."
    except Exception as e:
        return "Sorry, I'm having trouble connecting. Please try again."

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
    response = query_gemini(context, user_message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
