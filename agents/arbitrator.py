"""
arbitrator.py
Voting Arbitrator — the quality-control agent of ResearchMind.

IMPORTANT: This agent does NOT use an LLM.
It is pure Python logic (as defined in the original architecture).

Current behaviour  (1 researcher):
  → Checks the confidence score of the single ResearcherOutput.
  → If confidence >= threshold  →  ACCEPT.
  → If confidence <  threshold  →  REJECT (triggers a retry via LangGraph).

Future behaviour  (3 researchers):
  → Receives 3 ResearcherOutputs for the same subquery.
  → Uses majority voting: picks the answer that 2+ researchers agree on.
  → Compares semantic similarity of summaries to break ties.
  → Selects the best output based on combined confidence + agreement.

Why no LLM here?
  → The Arbitrator's job is objective evaluation, not creative generation.
  → Using an LLM would add latency, cost, and non-determinism.
  → Simple rules (confidence thresholds, voting) are faster and 100% reproducible.
"""
from core.schemas import ResearcherOutput, ArbitratorVerdict
from typing import List


# ── Configuration ──
# Minimum confidence score to accept a researcher's answer.
# Anything below this triggers a retry loop in LangGraph.
CONFIDENCE_THRESHOLD = 0.6


def evaluate_outputs(subquery: str, outputs: List[ResearcherOutput]) -> ArbitratorVerdict:
    """
    Evaluates a list of ResearcherOutputs for a single subquery.
    
    Right now with 1 researcher, this simply checks the confidence score.
    When we add Researcher_2 and Researcher_3, this function will be
    extended with majority voting and similarity scoring.
    
    Args:
        subquery:  The original sub-question.
        outputs:   List of ResearcherOutput objects (currently just 1).
    
    Returns:
        ArbitratorVerdict with accepted=True/False and the best output.
    """

    # ── Edge case: no outputs received ──
    if not outputs:
        return ArbitratorVerdict(
            subquery=subquery,
            accepted=False,
            reason="No researcher outputs received.",
            best_output=None
        )

    # ── Single researcher logic (current phase) ──
    # Pick the output with the highest confidence (future-proofed for 3 researchers)
    best = max(outputs, key=lambda x: x.confidence)

    if best.confidence >= CONFIDENCE_THRESHOLD:
        return ArbitratorVerdict(
            subquery=subquery,
            accepted=True,
            reason=f"Confidence {best.confidence:.2f} meets threshold {CONFIDENCE_THRESHOLD}.",
            best_output=best
        )
    else:
        return ArbitratorVerdict(
            subquery=subquery,
            accepted=False,
            reason=f"Confidence {best.confidence:.2f} is below threshold {CONFIDENCE_THRESHOLD}. Needs retry.",
            best_output=best
        )
