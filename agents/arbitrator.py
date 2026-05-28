"""
arbitrator.py
Voting Arbitrator — the quality-control agent of ResearchMind.

IMPORTANT: This agent does NOT use an LLM.
It is pure Python logic (as defined in the original architecture).

How Voting Works (with 3 researchers):
  1. Receive 3 ResearcherOutputs for the SAME subquery.
  2. Check if any outputs failed (confidence == 0.0).
  3. Calculate an average confidence score across all valid outputs.
  4. Pick the "best" output — highest individual confidence.
  5. If average confidence >= threshold → ACCEPT the best output.
  6. If average confidence <  threshold → REJECT (trigger retry via LangGraph).

Why no LLM here?
  → The Arbitrator's job is objective evaluation, not creative generation.
  → Using an LLM would add latency, cost, and non-determinism.
  → Simple rules (confidence thresholds, averaging) are faster and 100% reproducible.

Future enhancement ideas:
  → Semantic similarity between summaries (do researchers agree?)
  → Weighted voting based on source overlap
  → Penalise outputs that are too short or too generic
"""
from core.schemas import ResearcherOutput, ArbitratorVerdict
from typing import List


# ── Configuration ──
# Minimum AVERAGE confidence across all researchers to accept an answer.
CONFIDENCE_THRESHOLD = 0.6


def evaluate_outputs(subquery: str, outputs: List[ResearcherOutput], requires_quantitative: bool = False) -> ArbitratorVerdict:
    """
    Evaluates a list of ResearcherOutputs for a single subquery using voting logic.
    
    With 3 researchers, this performs majority-style voting:
      - Applies objective penalty rules (source count, length, digits) to self-reported scores.
      - Filters out failed outputs (confidence == 0.0)
      - Computes average confidence across valid outputs
      - Selects the highest-confidence output as "best"
      - Accepts if average confidence meets the threshold
    
    Args:
        subquery:              The original sub-question.
        outputs:               List of ResearcherOutput objects.
        requires_quantitative: True if this topic requires numbers and metrics.
    
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

    # ── Step 0: Apply objective quality checks to penalize confidence scores ──
    for i, o in enumerate(outputs, 1):
        if o.confidence <= 0.0:
            continue
            
        original_conf = o.confidence
        penalized = False
        reasons = []

        # 1. Source Citation Check (No Sources = No Trust)
        if not o.sources or len(o.sources) == 0:
            o.confidence = round(o.confidence * 0.2, 2)
            penalized = True
            reasons.append("cites 0 sources")

        # 2. Summary Length Check (Too Short)
        if len(o.summary) < 200:
            o.confidence = round(o.confidence * 0.5, 2)
            penalized = True
            reasons.append("shallow summary (< 200 chars)")

        # 3. Quantitative Data Check (For Hard Science)
        if requires_quantitative:
            has_numbers = any(char.isdigit() for char in o.summary)
            if not has_numbers:
                o.confidence = round(o.confidence * 0.5, 2)
                penalized = True
                reasons.append("missing quantitative data/numbers")

        if penalized:
            print(f"  [Arbitrator] ⚠️ Researcher {i} confidence penalized: {original_conf:.2f} ➔ {o.confidence:.2f} (Reasons: {', '.join(reasons)})")

    # ── Step 1: Filter out failed outputs (confidence == 0.0 means parse failure or heavy penalties) ──
    valid_outputs = [o for o in outputs if o.confidence > 0.0]
    total = len(outputs)
    valid = len(valid_outputs)

    print(f"  [Arbitrator] Received {total} outputs, {valid} valid.")

    if not valid_outputs:
        return ArbitratorVerdict(
            subquery=subquery,
            accepted=False,
            reason=f"All {total} researcher outputs failed quality checks or parsing.",
            best_output=None
        )

    # ── Step 2: Calculate average confidence (the "vote") ──
    avg_confidence = sum(o.confidence for o in valid_outputs) / valid

    # ── Step 3: Pick the best individual output ──
    best = max(valid_outputs, key=lambda x: x.confidence)

    # ── Step 4: Log the voting breakdown ──
    print(f"  [Arbitrator] Confidence scores: {[o.confidence for o in outputs]}")
    print(f"  [Arbitrator] Average confidence: {avg_confidence:.2f}")
    print(f"  [Arbitrator] Best individual confidence: {best.confidence:.2f}")

    # ── Step 5: Accept or reject based on average ──
    if avg_confidence >= CONFIDENCE_THRESHOLD:
        return ArbitratorVerdict(
            subquery=subquery,
            accepted=True,
            reason=(
                f"Average confidence {avg_confidence:.2f} meets threshold {CONFIDENCE_THRESHOLD}. "
                f"Best output (confidence {best.confidence:.2f}) selected from {valid}/{total} valid researchers."
            ),
            best_output=best
        )
    else:
        return ArbitratorVerdict(
            subquery=subquery,
            accepted=False,
            reason=(
                f"Average confidence {avg_confidence:.2f} is below threshold {CONFIDENCE_THRESHOLD}. "
                f"Best was {best.confidence:.2f}. Needs retry."
            ),
            best_output=best
        )
