#!/usr/bin/env python3
"""
LLM_QA_CLI.py
Simple CLI that accepts a question, preprocesses it, constructs a prompt and calls an LLM API (Gemini).
Uses GEMINI_API_KEY set in the environment. If the key or package is missing a helpful fallback message is shown.
"""
import os
import re
import sys

try:
    import google.generativeai as genai
except Exception:
    genai = None


def preprocess(text: str):
    """Lowercase, remove punctuation and tokenize."""
    text = text.strip().lower()
    # remove punctuation (keep word and whitespace characters)
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
        raise RuntimeError("google-generativeai package not installed. Install with: pip install google-generativeai")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    response = model.generate_content(prompt)
    return response.text.strip()


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")

    processed, tokens = preprocess(question)
    prompt = construct_prompt(processed)

    try:
        answer = call_gemini(prompt)
    except Exception as e:
        answer = (
            f"[LLM call failed: {e}]\n"
            "As a fallback, returning processed question tokens.\n"
            f"Processed: {processed}\nTokens: {tokens}"
        )

    print("\nProcessed question:\n", processed)
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
