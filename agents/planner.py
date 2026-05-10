"""
planner.py
Contains the LLM prompt and parsing logic to decompose 
a broad research question into exactly 3 focused sub-questions.
"""
import json
from utils.llm_client import call_groq_llm

def generate_subqueries(user_query: str) -> list[str]:
    """Takes a user query and returns a list of 3 subqueries."""
    system_prompt = """
    You are an expert research planner.
    Your goal is to break down a user's broad research query into exactly 3 focused sub-queries.

    Output your response strictly as a JSON object with a single key "subqueries" 
    that contains a list of exactly 3 strings.

    Example output:
    {
      "subqueries": ["subquery 1", "subquery 2", "subquery 3"]
    }
    """
    
    # 1. Get the raw JSON string response from the LLM
    raw_response = call_groq_llm(system_prompt=system_prompt, user_prompt=user_query)
    
    # 2. Parse the string into a Python dictionary
    parsed_data = json.loads(raw_response)
    
    # 3. Extract the list of strings and return it
    return parsed_data.get("subqueries", [])
