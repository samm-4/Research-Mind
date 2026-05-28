"""
synthesizer.py
Combined Synthesizer Agent for ResearchMind.

Takes the research summaries of all three subqueries and compiles them
into a single master research dossier containing a Unified Executive Summary,
detailed section-by-section reports, and a verified bibliography.
"""
import json
from typing import List
from core.schemas import ResearcherOutput
from utils.llm_client import call_gemini_llm


def synthesize_all_results(user_query: str, results: List[ResearcherOutput]) -> str:
    """
    Runs a single combined synthesis over all subquery results.
    Generates a unified master Markdown report using Gemini 2.5 Flash (with fallbacks).
    """
    print(f"\n  [Synthesizer] Compiling master research report for: '{user_query}'...")
    
    # 1. Collect all unique verified sources
    unique_sources = set()
    for r in results:
        if r.sources:
            for src in r.sources:
                unique_sources.add(src)
                
    # 2. Build the context of the subquery reports for the LLM
    dossier_context = ""
    for i, r in enumerate(results, 1):
        dossier_context += f"### Subquery {i}: {r.subquery}\n"
        dossier_context += f"Researcher Summary:\n{r.summary}\n"
        dossier_context += f"Sources Cited: {', '.join(r.sources) if r.sources else 'None'}\n\n"
        
    system_prompt = f"""
    You are an expert scientific editor and research director.
    Your task is to take a collection of research summaries for 3 different subqueries under the main topic: "{user_query}", and compile them into a single, cohesive, publication-grade research dossier.
    
    Avoid writing long, essay-like prose. Instead, focus on structure and precision.
    
    Your output MUST be structured in clean Markdown exactly as follows:

    # Comprehensive Research Dossier: {user_query}

    ## Executive Summary & Key Findings
    - Synthesize a high-level, overarching executive summary of the entire research topic.
    - Weave together the insights from all three subqueries into a single unified picture.
    - Highlight the main consensus points, active scientific debates, and overall conclusions.

    ---

    ## Detailed Subquery Breakdowns

    ### 1. {results[0].subquery if len(results) > 0 else 'Subquery 1'}
    
    #### Consensus & Benchmarks
    - Summarize SOTA benchmarks, numbers, and facts.
    - Highlight specific quantitative metrics and chemistry/material names.
    - Cite sources using bracketed numbers corresponding to the sources in this section.

    #### Scientific Contradictions & Debates
    - Outline disputes, competing methodologies, or conflicting data in the literature.

    #### Key Data Gaps
    - Identify what critical technical details, materials parameters, or scaling challenges are missing or left unaddressed.

    ---

    ### 2. {results[1].subquery if len(results) > 1 else 'Subquery 2'}
    
    #### Consensus & Benchmarks
    - Summarize SOTA benchmarks, numbers, and facts.
    - Highlight specific quantitative metrics and chemistry/material names.
    
    #### Scientific Contradictions & Debates
    - Outline disputes, competing methodologies, or conflicting data in the literature.

    #### Key Data Gaps
    - Identify what critical technical details, materials parameters, or scaling challenges are missing or left unaddressed.

    ---

    ### 3. {results[2].subquery if len(results) > 2 else 'Subquery 3'}
    
    #### Consensus & Benchmarks
    - Summarize SOTA benchmarks, numbers, and facts.
    - Highlight specific quantitative metrics and chemistry/material names.
    
    #### Scientific Contradictions & Debates
    - Outline disputes, competing methodologies, or conflicting data in the literature.

    #### Key Data Gaps
    - Identify what critical technical details, materials parameters, or scaling challenges are missing or left unaddressed.
    """

    user_prompt = f"Main Topic: {user_query}\n\nSubquery Findings:\n{dossier_context}"
    
    try:
        # Call Gemini (utilizes retry-after sleep logic and falls back to Groq Llama if needed)
        report = call_gemini_llm(system_prompt=system_prompt, user_prompt=user_prompt, json_mode=False)
    except Exception as e:
        print(f"  [Synthesizer Error] Failed to generate master report: {e}")
        # Build a basic fallback report using python text concatenation
        report = f"# Comprehensive Research Dossier: {user_query}\n\n## Executive Summary\n- Automated synthesis failed due to API connection issues.\n\n## Detailed Results\n"
        for r in results:
            report += f"\n### {r.subquery}\n- {r.summary}\n"
            
    # Append the master bibliography list at the very end
    report += "\n\n## 📚 Master Bibliography\n"
    if unique_sources:
        for src in sorted(unique_sources):
            report += f"- {src}\n"
    else:
        report += "- No sources cited.\n"
        
    return report
