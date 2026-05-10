"""
researcher.py
Logic for Researcher_1 agent.

This is the core "knowledge worker" of the pipeline.
For each subquery from the Planner, it:
  1. Searches the web using Tavily  (Retrieval step of RAG)
  2. Injects the search results into a prompt  (Augmentation step)
  3. Asks Gemini 2.0 Flash to summarise the findings  (Generation step)
  4. Returns a validated ResearcherOutput via Pydantic

Why Gemini for the Researcher?
  → Gemini 2.0 Flash has a 1M-token context window.
  → The Researcher feeds in large blocks of web-search context,
    so a big context window is essential.
  → It's free via Google AI Studio.
"""
import json
from pydantic import ValidationError
from tools.search import perform_web_search
from utils.llm_client import call_gemini_llm
from core.schemas import ResearcherOutput


def research_subquery(subquery: str) -> ResearcherOutput:
    """
    Full RAG pipeline for a single subquery.
    Returns a validated ResearcherOutput (subquery, summary, sources, confidence).
    """

    # ── Step 1: RETRIEVE — Search the web for live information ──
    print(f"  [Researcher] Searching web for: '{subquery}'")
    context = perform_web_search(subquery)

    # ── Step 2: AUGMENT — Build the prompt with the retrieved context ──
    system_prompt = """
    You are an expert web researcher.
    You will be provided with a subquery and context from web searches.
    Your task is to summarize the information found in the context that answers the subquery.
    
    Output your response strictly as a JSON object matching the following structure:
    {
      "subquery": "The original subquery",
      "summary": "Your detailed summary based on the context",
      "sources": ["URL 1", "URL 2"],
      "confidence": 0.95
    }
    
    Rules:
    - Only use information from the provided context.
    - Extract the actual URLs from the context and place them in "sources".
    - If the context does not contain enough information, say so in the summary
      and lower the confidence score accordingly.
    """

    user_prompt = f"Subquery: {subquery}\n\nContext:\n{context}"

    # ── Step 3: GENERATE — Ask Gemini to produce a structured answer ──
    print(f"  [Researcher] Asking Gemini to summarise findings...")
    raw_response = call_gemini_llm(system_prompt=system_prompt, user_prompt=user_prompt)

    # ── Step 4: VALIDATE — Parse + validate through Pydantic ──
    try:
        parsed_data = json.loads(raw_response)
        output = ResearcherOutput(**parsed_data)
        return output
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"  [Researcher Error] Failed to parse LLM response: {e}")
        # Return a safe fallback so the pipeline doesn't crash
        return ResearcherOutput(
            subquery=subquery,
            summary="Failed to parse LLM response.",
            sources=[],
            confidence=0.0
        )
