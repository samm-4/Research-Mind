"""
schemas.py
Defines the strict JSON structures using Pydantic.

These schemas serve two purposes:
  1. They validate LLM outputs so malformed JSON doesn't crash the pipeline.
  2. They act as "contracts" between agents — each agent knows exactly
     what shape of data it will receive and must produce.
"""
from pydantic import BaseModel
from typing import List, Optional


class ResearcherOutput(BaseModel):
    """
    The structured output from a single Researcher agent.
    
    Fields:
      - subquery:    The original sub-question this result answers.
      - summary:     The LLM's summary based on the web search context.
      - sources:     List of URLs the summary was derived from.
      - confidence:  A 0.0–1.0 score indicating how well the context
                     answered the subquery (set by the LLM).
    """
    subquery: str
    summary: str
    sources: List[str]
    confidence: float


class ArbitratorVerdict(BaseModel):
    """
    The output of the Voting Arbitrator for a single subquery.
    
    The Arbitrator inspects the ResearcherOutput(s) and decides
    whether the answer is good enough or needs a retry.
    
    Fields:
      - subquery:       The subquery being evaluated.
      - accepted:       True if the answer passed quality checks.
      - reason:         Why it was accepted or rejected.
      - best_output:    The ResearcherOutput that was selected as best
                        (right now there's only 1 researcher, but this
                         scales to 3 researchers with majority voting).
    """
    subquery: str
    accepted: bool
    reason: str
    best_output: Optional[ResearcherOutput] = None
