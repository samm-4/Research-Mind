"""
llm_client.py
Centralised LLM interface for ResearchMind.

We use TWO different models for different jobs:
  • Groq  (Llama 3.3 70B)   → Planner agent  (fast, great at structured reasoning)
  • Gemini (2.0 Flash)       → Researcher agent (1 M-token context window for large web results)

Each function accepts a system prompt and user prompt and returns
the raw text string from the model's response.
"""
import os
from groq import Groq
from google import genai


# ──────────────────────────────────────────────
# 1.  GROQ  –  Used by the Planner
# ──────────────────────────────────────────────
def call_groq_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Calls Groq's Llama-3.3-70B model.
    Returns the raw text/JSON string from the model.
    
    Why Groq for Planner?
      - Groq is extremely fast (runs on custom LPU hardware).
      - Llama 3.3 70B is excellent at strict JSON adherence.
      - The Planner's prompt is short, so we don't need a huge context window.
    """
    # The client automatically reads GROQ_API_KEY from the environment
    client = Groq()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,                        # Deterministic output
        response_format={"type": "json_object"}  # Forces valid JSON
    )

    return response.choices[0].message.content


# ──────────────────────────────────────────────
# 2.  GEMINI  –  Used by the Researcher
# ──────────────────────────────────────────────
def call_gemini_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Calls Google's Gemini 2.5 Flash model.
    Returns the raw text/JSON string from the model.
    
    Why Gemini for the Researcher?
      - Gemini 2.5 Flash has a 1 million token context window.
      - The Researcher feeds in large blocks of web-search context,
        so a big context window is essential.
      - It's free via Google AI Studio.
    """
    # The client reads GEMINI_API_KEY from the environment
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{system_prompt}\n\n{user_prompt}",
        config={
            "response_mime_type": "application/json",  # Forces valid JSON output
            "temperature": 0.0,                        # Deterministic output
        }
    )

    return response.text
