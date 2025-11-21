from flask import Flask, render_template, request, jsonify
import os
import re

try:
    import google.generativeai as genai
except Exception:
    genai = None

app = Flask(__name__)


def preprocess(text: str):
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return text, tokens


def construct_prompt(processed_question: str) -> str:
    return (
        "You are an expert assistant. Answer concisely and clearly.\n"
        f"Question: {processed_question}\n"
        "Answer:"
    )


def call_gemini(prompt: str) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai package not installed")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    response = model.generate_content(prompt)
    return response.text.strip()


@app.route("/", methods=["GET"]) 
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "")
    processed, tokens = preprocess(question)
    prompt = construct_prompt(processed)
    try:
        answer = call_gemini(prompt)
    except Exception as e:
        answer = f"[LLM call failed: {e}]\nProcessed: {processed}\nTokens: {tokens}"
    return jsonify({"processed": processed, "answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
