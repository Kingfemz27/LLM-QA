#!/usr/bin/env python3
"""Lightweight Flask web app that forwards user questions to a generative LLM.
Refactored for clarity and to avoid direct copying of the original implementation.
"""

import os
import re
import logging
from typing import Optional
from dotenv import load_dotenv
from flask import Flask, request, render_template
from google import genai

# Initialize environment and logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "gemini-2.5-flash"
SYSTEM_PROMPT = (
    "You are a concise and factual question-answering assistant. "
    "Answer clearly and directly to the user query."
)

app = Flask(__name__)


def normalize_text(s: Optional[str]) -> str:
    """Normalize input text by trimming, lowercasing, and removing punctuation.
    Keeps only basic word characters and spaces.
    """
    if not s:
        return ""
    s = s.strip().lower()
    # remove punctuation and any non-word characters except spaces
    s = re.sub(r"[^\w\s]", "", s)
    # collapse extra whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def call_llm(question: str) -> str:
    """Send question to the LLM and return the text response. Handles errors gracefully."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("LLM API key is not set in environment")

    client = genai.Client(api_key=api_key)

    # Build the prompt content in the expected format
    prompt = f"Question: {question}"

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"system_instruction": SYSTEM_PROMPT},
    )

    # genai response object exposes the text via .text (best-effort)
    return getattr(resp, "text", "")


@app.route("/", methods=["GET", "POST"])
def home():
    original = ""
    normalized = ""
    answer = ""
    error_msg = ""

    if request.method == "POST":
        original = request.form.get("question", "").strip()
        if not original:
            error_msg = "Please provide a question before submitting."
            logger.debug("Empty question submitted")
            return render_template("index.html", error=error_msg)

        normalized = normalize_text(original)

        try:
            answer = call_llm(normalized)
        except Exception as ex:
            logger.exception("Error calling LLM")
            error_msg = f"An internal error occurred while contacting the LLM: {ex}"

    return render_template(
        "index.html",
        question=original,
        processed_question=normalized,
        llm_response=answer,
        error=error_msg,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug controlled via environment for deployment safety
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)