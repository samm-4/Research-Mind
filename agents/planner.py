"""
planner.py
Contains the LLM prompt and parsing logic to decompose 
a broad research question into exactly 3 focused sub-questions.
"""
import json
from utils.llm_client import call_groq_llm

def generate_subqueries(user_query: str) -> dict:
    """Takes a user query and returns a dictionary with 'subqueries' and 'requires_quantitative_metrics'."""
    system_prompt = """
    You are an expert academic research planner.
    Your goal is to break down a user's broad research query into exactly 3 distinct, concise, and focused sub-queries.
    
    First, determine if answering this query requires extracting quantitative metrics/numerical data to be valid and scientific (e.g., chemistry, physics, biology/clinical trials, engineering).
    Set "requires_quantitative_metrics" to true if numbers, percentages, efficiency rates, clinical trial stats, or physical/chemical measurements are essential. Otherwise, set it to false.
    
    Next, generate exactly 3 sub-queries that break down the topic logically. Do NOT use rigid templates or repeat the main query word-for-word. Instead, adapt the sub-queries naturally to the topic's domain.
    
    Structure the 3 sub-queries around these three dimensions:
    1. Efficacy, Performance, or Core Results: What does the data/literature show about the primary effect or performance of the options?
    2. Underlying Mechanisms, Theories, or Science: How or why does it work? What are the physical, chemical, biological, or architectural processes involved?
    3. Practical Constraints, Drawbacks, or Implementation Limits: What are the real-world limitations? (e.g., side effects and compliance for medicine; cost and scalability for engineering; ethics and adoption barriers for technology).
    
    Keep the sub-queries concise, search-friendly, and natural. Avoid overly verbose preamble phrases (like "What are the current state-of-the-art findings on...").

    Output your response strictly as a JSON object with two keys:
    {
      "subqueries": ["concise subquery 1", "concise subquery 2", "concise subquery 3"],
      "requires_quantitative_metrics": true or false
    }

    Example output format:
    {
      "subqueries": [
        "Clinical efficacy of intermittent fasting vs continuous calorie restriction on insulin sensitivity in type 2 diabetes.",
        "Physiological mechanisms of how fasting and calorie restriction influence glucose metabolism and beta-cell function.",
        "Adherence rates, compliance challenges, and clinical side effects of intermittent fasting vs calorie restriction."
      ],
      "requires_quantitative_metrics": true
    }
    """
    
    # 1. Get the raw JSON string response from the LLM
    raw_response = call_groq_llm(system_prompt=system_prompt, user_prompt=user_query)
    
    # 2. Parse the string into a Python dictionary
    parsed_data = json.loads(raw_response)
    
    # 3. Return the parsed dictionary
    return parsed_data
