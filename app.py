from flask import Flask, render_template, request, jsonify
import json
import requests
import numpy as np
import os

app = Flask(__name__)

# HuggingFace API - Free, No RAM usage
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-albert-small-v2"
HF_TOKEN = os.getenv("HF_TOKEN") # Render me baad me add karenge

def get_embedding(text):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text})
    return np.array(response.json())

# Load FAQs
with open('faqs.json', 'r') as f:
    faqs = json.load(f)

# Pre-compute embeddings once at startup using API
faq_questions = [item['question'] for item in faqs]
print("Computing embeddings via HuggingFace API...")
faq_embeddings = np.array([get_embedding(q) for q in faq_questions])
print("Embeddings ready!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json['question']
    user_embedding = get_embedding(user_question)
    
    # Cosine similarity
    similarities = np.dot(faq_embeddings, user_embedding) / (
        np.linalg.norm(faq_embeddings, axis=1) * np.linalg.norm(user_embedding)
    )
    best_match_idx = np.argmax(similarities)
    
    if similarities[best_match_idx] > 0.6:
        answer = faqs[best_match_idx]['answer']
    else:
        answer = "Sorry, main is sawaal ka jawab nahi de paayi. Kripya MindBrain ke website pe contact karein."
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
