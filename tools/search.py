"""
search.py
Isolates the web search logic using the Tavily API.
"""
import os
from tavily import TavilyClient

def perform_web_search(query: str, max_results: int = 3) -> str:
    """
    Searches the web using Tavily and returns a clean, combined string of the text.
    """
    # 1. Initialize the client (it automatically looks for TAVILY_API_KEY in the OS)
    client = TavilyClient()
    
    # 2. Perform the search. 'basic' is fast and uses fewer credits.
    response = client.search(
        query=query, 
        max_results=max_results, 
        search_depth="basic"
    )
    
    # 3. Combine the results into a single string
    context = ""
    for result in response.get("results", []):
        context += f"Source: {result.get('url')}\n"
        context += f"Content: {result.get('content')}\n\n"
        
    return context.strip()

