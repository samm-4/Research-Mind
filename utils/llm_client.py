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
import re
import time
from groq import Groq
from google import genai


# ──────────────────────────────────────────────
# 1.  GROQ  –  Used by the Planner
# ──────────────────────────────────────────────
def call_groq_llm(system_prompt: str, user_prompt: str, model: str = "llama-3.3-70b-versatile", json_mode: bool = True) -> str:
    """
    Calls a model via Groq (defaults to Llama-3.3-70B).
    Falls back to Llama-3.1-8B-instant if Llama-3.3 is exhausted or rate-limited.
    """
    try:
        # The client automatically reads GROQ_API_KEY from the environment
        client = Groq()

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        # If Llama-3.3 is rate-limited or out of tokens, fall back to Llama-3.1-8B
        if model == "llama-3.3-70b-versatile":
            print(f"\n  ⚠️ [Groq Error] Llama-3.3 limit reached ({e}). Falling back to Llama-3.1-8B...")
            return call_groq_llm(system_prompt, user_prompt, model="llama-3.1-8b-instant", json_mode=json_mode)
        else:
            raise e


# ──────────────────────────────────────────────
# 2.  GEMINI  –  Used by the Researcher
# ──────────────────────────────────────────────
def call_gemini_llm(system_prompt: str, user_prompt: str, json_mode: bool = True, retry_count: int = 0) -> str:
    """
    Calls Google's Gemini 2.5 Flash model.
    If Gemini hits rate limits (429), it parses the retry delay, sleeps, and retries.
    Falls back to Groq if retries are exhausted or it gets other exceptions.
    """
    try:
        # The client reads GEMINI_API_KEY from the environment
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        config = {
            "temperature": 0.0
        }
        if json_mode:
            config["response_mime_type"] = "application/json"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{system_prompt}\n\n{user_prompt}",
            config=config
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        
        # Check if it is a rate limit / quota exhaustion (429) error
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            # Attempt to extract the retry delay from the error message
            delay = 5.0  # Default fallback sleep time
            match = re.search(r"Please retry in ([\d\.]+)s", error_msg)
            if match:
                delay = float(match.group(1))
            
            # Option 3 (Balanced Pacing): Only sleep and retry if the wait time is short (<= 10s)
            # If the wait time is long, immediately fall back to Groq/Llama to avoid slowing the user down.
            if delay <= 10.0 and retry_count < 1:
                print(f"\n  ⚠️ [Gemini Rate Limit] Exceeded quota. Sleeping for {delay:.2f}s before retry 1/1...")
                time.sleep(delay + 0.5)  # Add a small safety buffer
                return call_gemini_llm(system_prompt, user_prompt, json_mode=json_mode, retry_count=retry_count + 1)
            else:
                print(f"\n  ⚠️ [Gemini Rate Limit] Wait time is too long ({delay:.2f}s). Skipping wait and falling back to Groq/Llama...")

        # For other errors, or if retries are exhausted/skipped, fall back to Groq/Llama
        print(f"\n  ⚠️ [Gemini Error] Service unavailable ({error_msg}). Falling back to Groq/Llama...")
        return call_groq_llm(system_prompt, user_prompt, json_mode=json_mode)
