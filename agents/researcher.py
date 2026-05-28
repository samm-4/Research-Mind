"""
researcher.py
Contains ALL researcher agents for ResearchMind.

We use 3 independent researchers to get diverse perspectives on
the same subquery. Each researcher differs in either:
  - The LLM used (Gemini vs Groq)
  - The search depth / number of results
  - The prompt style (analytical vs comprehensive vs critical)

This diversity is essential for the Voting Arbitrator to work —
if all 3 gave identical answers, voting would be pointless.

Researcher Map:
  ┌──────────────┬────────────────────┬─────────────────────┬──────────────────┐
  │  Researcher  │  LLM              │  Search Strategy    │  Prompt Style    │
  ├──────────────┼────────────────────┼─────────────────────┼──────────────────┤
  │  #1          │  Gemini 2.5 Flash  │  3 results, academic│  Balanced        │
  │  #2          │  Groq (Llama 3.3)  │  3 results, academic│  Analytical      │
  │  #3          │  Groq (Llama-8B)   │  5 results, academic│  Critical        │
  └──────────────┴────────────────────┴─────────────────────┴──────────────────┘
"""
import json
from pydantic import ValidationError
from tools.search import perform_web_search
from utils.llm_client import call_gemini_llm, call_groq_llm
from core.schemas import ResearcherOutput


# ══════════════════════════════════════════════
#  RESEARCHER 1 — Gemini 2.5 Flash (Balanced)
# ══════════════════════════════════════════════
def research_subquery_1(subquery: str, requires_quantitative: bool = False) -> ResearcherOutput:
    """
    Researcher_1: Uses Gemini + academic search + balanced prompt.
    This is the "generalist" — aims for a well-rounded summary.
    """
    print(f"  [Researcher 1] Searching web for: '{subquery}'")
    context = perform_web_search(subquery, max_results=3, academic=True)

    if requires_quantitative:
        metrics_instruction = (
            "- You MUST search for and extract quantitative data (specific metrics, numbers, units like kJ/mol, kWh/ton, USD/ton) and specific chemical/materials names (e.g. mmen-Mg2(dobpdc), UiO-66).\n"
            "- Avoid vague generalizations like 'high efficiency,' 'lower costs,' or 'improved stability.' If the context states 'high capacity,' look for the exact capacity value (e.g., '3.0 mmol/g').\n"
            "- If the context completely lacks quantitative data or concrete materials names, state 'No quantitative data or materials found in search results' in your summary and set the confidence to 0.4."
        )
    else:
        metrics_instruction = (
            "- You should prioritize facts and figures if available, but you may also summarize qualitative findings (e.g. opinions, trends, surveys).\n"
            "- Do NOT penalize your confidence score or write 'No quantitative data found' just because raw physical or chemical numbers are missing."
        )

    system_prompt = f"""
    You are an expert web researcher.
    You will be provided with a subquery and context from web searches.
    Your task is to summarize the information found in the context that answers the subquery.
    
    Output your response strictly as a JSON object matching the following structure:
    {{
      "subquery": "The original subquery",
      "summary": "Your detailed summary based on the context (MUST be a plain text string, do NOT use a nested JSON object/dictionary)",
      "sources": ["URL 1", "URL 2"],
      "confidence": 0.95
    }}
    
    Rules:
    - The 'summary' field MUST be a string. Do NOT output a nested JSON structure inside 'summary'.
    - Only use information from the provided context.
    - Extract the actual URLs from the context and place them in "sources".
    {metrics_instruction}
    """

    user_prompt = f"Subquery: {subquery}\n\nContext:\n{context}"

    print(f"  [Researcher 1] Asking Gemini to summarise findings...")
    raw_response = call_gemini_llm(system_prompt=system_prompt, user_prompt=user_prompt)

    return _parse_response(raw_response, subquery, researcher_id=1, context=context)


# ══════════════════════════════════════════════
#  RESEARCHER 2 — Groq / Llama 3.3 (Analytical)
# ══════════════════════════════════════════════
def research_subquery_2(subquery: str, requires_quantitative: bool = False) -> ResearcherOutput:
    """
    Researcher_2: Uses Groq (Llama 3.3) + academic search + analytical prompt.
    This gives a DIFFERENT LLM's perspective on the same data.
    
    Why a different LLM?
      Different models have different reasoning biases and strengths.
      Llama 3.3 may catch details that Gemini misses, and vice versa.
      This diversity makes the Arbitrator's voting actually meaningful.
    """
    print(f"  [Researcher 2] Searching web for: '{subquery}'")
    context = perform_web_search(subquery, max_results=3, academic=True)

    if requires_quantitative:
        metrics_instruction = (
            "- Prioritize hard facts, numbers, thermodynamic values, capacity metrics, and cost data.\n"
            "- You MUST avoid vague qualitative descriptions (like 'high rate' or 'inexpensive'). Extract the exact numbers, units, and chemistry setups.\n"
            "- If the context lacks concrete data, state 'No quantitative data or materials found in search results' and set the confidence to 0.4."
        )
    else:
        metrics_instruction = (
            "- Prioritize facts, statistics, and concrete examples. Focus on solid evidence rather than general overviews.\n"
            "- Do NOT penalize your confidence score or write 'No quantitative data found' just because raw physical or chemical numbers are missing."
        )

    system_prompt = f"""
    You are an analytical research assistant. 
    You will be provided with a subquery and context from web searches.
    
    Your task is to analyze the context carefully and produce a fact-driven,
    structured summary. Focus on specific data points, statistics, and 
    concrete examples rather than general overviews.
    
    Output your response strictly as a JSON object matching the following structure:
    {{
      "subquery": "The original subquery",
      "summary": "Your analytical summary with specific facts and data points (MUST be a plain text string, do NOT use a nested JSON object/dictionary)",
      "sources": ["URL 1", "URL 2"],
      "confidence": 0.95
    }}
    
    Rules:
    - The 'summary' field MUST be a string. Do NOT output a nested JSON structure inside 'summary'.
    - Only use information from the provided context.
    - Extract the actual URLs from the context and place them in "sources".
    {metrics_instruction}
    """

    user_prompt = f"Subquery: {subquery}\n\nContext:\n{context}"

    print(f"  [Researcher 2] Asking Groq/Llama to analyse findings...")
    raw_response = call_groq_llm(system_prompt=system_prompt, user_prompt=user_prompt)

    return _parse_response(raw_response, subquery, researcher_id=2, context=context)


# ══════════════════════════════════════════════
#  RESEARCHER 3 — Gemini 2.5 Flash (Critical + Deep Search)
# ══════════════════════════════════════════════
def research_subquery_3(subquery: str, requires_quantitative: bool = False) -> ResearcherOutput:
    """
    Researcher_3: Uses Groq (Llama-3.1-8B) + DEEPER academic search (5 results) + critical prompt.
    
    This researcher gets MORE data (5 results instead of 3)
    and is prompted to be critical — questioning claims, noting gaps,
    and flagging potential biases. This acts as a "devil's advocate".
    """
    print(f"  [Researcher 3] Deep searching web for: '{subquery}'")
    context = perform_web_search(subquery, max_results=5, academic=True)

    if requires_quantitative:
        metrics_instruction = (
            "- Actively search for and point out contradictory metrics or numbers in different papers (e.g. differing cost projections, contradictory degradation rates).\n"
            "- Flag assertions that are qualitative marketing hype and lack concrete quantitative numbers.\n"
            "- If evidence is weak or contradictory, lower the confidence score."
        )
    else:
        metrics_instruction = (
            "- Actively point out any consensus or debates in different papers, highlighting conflicting approaches, opinions, or studies.\n"
            "- Focus on evaluating critical gaps in evidence without penalizing for a lack of hard physics/chemical numbers."
        )

    system_prompt = f"""
    You are a critical research reviewer.
    You will be provided with a subquery and context from web searches.
    
    Your task is to critically evaluate the context and produce a thorough summary.
    Look for consensus across sources, note contradictions, and flag any 
    claims that lack strong evidence.
    
    Output your response strictly as a JSON object matching the following structure:
    {{
      "subquery": "The original subquery",
      "summary": "Your critical summary noting consensus, contradictions, and evidence gaps (MUST be a plain text string, do NOT use a nested JSON object/dictionary)",
      "sources": ["URL 1", "URL 2", "URL 3"],
      "confidence": 0.95
    }}
    
    Rules:
    - The 'summary' field MUST be a string. Do NOT output a nested JSON structure inside 'summary'.
    - Only use information from the provided context.
    - Extract the actual URLs from the context and place them in "sources".
    {metrics_instruction}
    """

    user_prompt = f"Subquery: {subquery}\n\nContext:\n{context}"

    print(f"  [Researcher 3] Asking Groq/Llama-8B to critically evaluate findings...")
    raw_response = call_groq_llm(system_prompt=system_prompt, user_prompt=user_prompt, model="llama-3.1-8b-instant")

    return _parse_response(raw_response, subquery, researcher_id=3, context=context)


# ══════════════════════════════════════════════
#  SHARED HELPER — Parses LLM response into ResearcherOutput
# ══════════════════════════════════════════════
def _parse_response(raw_response: str, subquery: str, researcher_id: int, context: str) -> ResearcherOutput:
    """
    Shared parsing logic used by all 3 researchers.
    Validates the JSON response through Pydantic, filters out hallucinated sources, and returns a safe fallback on failure.
    """
    try:
        parsed_data = json.loads(raw_response)
        output = ResearcherOutput(**parsed_data)
        
        # Verify citations: only keep URLs that exist in the raw search context
        verified_sources = []
        for url in output.sources:
            if url in context:
                verified_sources.append(url)
            else:
                print(f"  [Researcher {researcher_id}] ⚠️ Removed unverified/hallucinated source URL: {url}")
        output.sources = verified_sources
        
        return output
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"  [Researcher {researcher_id} Error] Failed to parse LLM response: {e}")
        return ResearcherOutput(
            subquery=subquery,
            summary="Failed to parse LLM response.",
            sources=[],
            confidence=0.0
        )
