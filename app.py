from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer # ✅ Line 2 FIX
import faiss
import numpy as np
import os

app = Flask(__name__)
hr_policy = [
    "MindBrain employees get 20 paid leaves per year.",
    "Work from home is allowed 2 days per week with manager approval.", 
    "Maternity leave is 26 weeks as per company policy.",
    "Provident Fund contribution is 12% from employee and employer both.",
    "Notice period during probation is 15 days, post probation is 60 days."
]

model = SentenceTransformer('paraphrase-albert-small-v2') # ✅ Line 16 FIX - Capital S
embeddings = model.encode(hr_policy)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST']) # ✅ '/query' ko '/ask' kar de
def ask():
    question = request.json['question']
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding), k=2)
    
    context = " ".join([hr_policy[i] for i in I[0]])
    answer = f"Based on HR Policy: {context}"
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
