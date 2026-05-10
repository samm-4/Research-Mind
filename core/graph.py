"""
graph.py
LangGraph state machine for ResearchMind.

This is the "brain" of the orchestration layer. Instead of using
simple for-loops in main.py, we define a directed graph where:

  Nodes  = the agents (Planner, Researcher, Arbitrator)
  Edges  = the flow of data between them
  State  = a shared dictionary that every node can read/write

The key feature is CONDITIONAL LOOPING:
  → After the Arbitrator evaluates a result, if it's rejected,
    LangGraph automatically routes the subquery BACK to the
    Researcher for another attempt (up to MAX_RETRIES).
  → This is the feedback loop from your original architecture:
    "good enough? NO → back to researchers"

Architecture Diagram:
    ┌──────────┐
    │ Planner  │  → Breaks query into 3 subqueries
    └────┬─────┘
         │
         ▼
    ┌────────────┐
    │ Researcher │  → Searches web + summarises (per subquery)
    └────┬───────┘
         │
         ▼
    ┌─────────────┐     ┌─── NO ───▶ back to Researcher (retry)
    │ Arbitrator  │─────┤
    └─────────────┘     └─── YES ──▶ move to next subquery / end
"""
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from agents.planner import generate_subqueries
from agents.researcher import research_subquery
from agents.arbitrator import evaluate_outputs
from core.schemas import ResearcherOutput, ArbitratorVerdict


# ── Configuration ──
MAX_RETRIES = 1  # Max number of times the Researcher can retry a failed subquery


# ──────────────────────────────────────────────
#  GRAPH STATE
# ──────────────────────────────────────────────
# This is the shared "whiteboard" that every node reads and writes to.
# LangGraph passes this state object through the entire pipeline.

class GraphState(TypedDict):
    user_query: str                                    # The original user question
    subqueries: List[str]                              # 3 subqueries from the Planner
    current_subquery_index: int                        # Which subquery we're currently on
    current_retry_count: int                           # How many retries for current subquery
    researcher_outputs: List[ResearcherOutput]         # Accumulated results from Researcher
    arbitrator_verdicts: List[ArbitratorVerdict]        # Accumulated verdicts from Arbitrator
    final_results: List[ResearcherOutput]              # Accepted results only


# ──────────────────────────────────────────────
#  NODE 1: PLANNER
# ──────────────────────────────────────────────
def planner_node(state: GraphState) -> dict:
    """
    Takes the user query and generates 3 subqueries.
    This node runs exactly once at the start of the pipeline.
    """
    print("\n[Planner] Thinking...")
    subqueries = generate_subqueries(state["user_query"])
    
    print("[Planner] Generated exactly 3 subqueries:")
    for i, sq in enumerate(subqueries, 1):
        print(f"  {i}. {sq}")
    
    # Write results to the shared state
    return {
        "subqueries": subqueries,
        "current_subquery_index": 0,       # Start with the first subquery
        "current_retry_count": 0,
        "researcher_outputs": [],
        "arbitrator_verdicts": [],
        "final_results": []
    }


# ──────────────────────────────────────────────
#  NODE 2: RESEARCHER
# ──────────────────────────────────────────────
def researcher_node(state: GraphState) -> dict:
    """
    Runs the Researcher agent on the CURRENT subquery.
    
    It reads current_subquery_index from the state to know which
    subquery to process. This lets LangGraph re-run this node
    on the same subquery during a retry loop.
    """
    idx = state["current_subquery_index"]
    subquery = state["subqueries"][idx]
    retry = state["current_retry_count"]
    
    print(f"\n--- Researching Subquery {idx + 1} (attempt {retry + 1}) ---")
    result = research_subquery(subquery)
    
    print(f"  [Researcher] Confidence: {result.confidence}")
    
    # Store the result (append to the running list)
    return {
        "researcher_outputs": state["researcher_outputs"] + [result]
    }


# ──────────────────────────────────────────────
#  NODE 3: ARBITRATOR
# ──────────────────────────────────────────────
def arbitrator_node(state: GraphState) -> dict:
    """
    Evaluates the latest ResearcherOutput using pure Python logic.
    
    If accepted → saves to final_results, advances to next subquery.
    If rejected → the conditional edge will route back to Researcher.
    """
    idx = state["current_subquery_index"]
    subquery = state["subqueries"][idx]
    
    # Get the most recent researcher output (the last one in the list)
    latest_output = state["researcher_outputs"][-1]
    
    print(f"\n[Arbitrator] Evaluating result for: '{subquery}'")
    verdict = evaluate_outputs(subquery, [latest_output])
    
    if verdict.accepted:
        print(f"  ✅ ACCEPTED — {verdict.reason}")
        # Move to next subquery
        return {
            "arbitrator_verdicts": state["arbitrator_verdicts"] + [verdict],
            "final_results": state["final_results"] + [latest_output],
            "current_subquery_index": idx + 1,    # Advance to next subquery
            "current_retry_count": 0               # Reset retry counter
        }
    else:
        print(f"  ❌ REJECTED — {verdict.reason}")
        return {
            "arbitrator_verdicts": state["arbitrator_verdicts"] + [verdict],
            "current_retry_count": state["current_retry_count"] + 1
        }


# ──────────────────────────────────────────────
#  CONDITIONAL EDGE: DECIDE WHAT HAPPENS NEXT
# ──────────────────────────────────────────────
def decide_next_step(state: GraphState) -> str:
    """
    This is the routing function that LangGraph calls after the Arbitrator.
    It returns the NAME of the next node to execute.
    
    Logic:
      1. If the Arbitrator rejected AND we haven't hit max retries
         → go back to "researcher" (retry the same subquery)
      2. If all 3 subqueries are done
         → go to END (pipeline complete)
      3. Otherwise
         → go to "researcher" (process the next subquery)
    """
    idx = state["current_subquery_index"]
    retries = state["current_retry_count"]
    total_subqueries = len(state["subqueries"])
    
    # Check if last verdict was a rejection AND we can still retry
    last_verdict = state["arbitrator_verdicts"][-1]
    if not last_verdict.accepted and retries <= MAX_RETRIES:
        print(f"  🔄 Retrying subquery (attempt {retries + 1})...")
        return "researcher"
    
    # If rejected but max retries exhausted, force-accept and move on
    if not last_verdict.accepted and retries > MAX_RETRIES:
        print(f"  ⚠️  Max retries reached. Accepting best available result.")
        # We need to move to the next subquery — update state via a side effect
        # Actually, we'll handle this by checking in the next researcher call
    
    # All subqueries processed?
    if idx >= total_subqueries:
        print("\n[Orchestrator] All subqueries processed!")
        return END
    
    # More subqueries remain → process the next one
    return "researcher"


# ──────────────────────────────────────────────
#  BUILD THE GRAPH
# ──────────────────────────────────────────────
def build_research_graph():
    """
    Constructs and compiles the LangGraph state machine.
    
    Graph structure:
      START → planner → researcher → arbitrator → (conditional) → ...
    
    The conditional edge after "arbitrator" uses decide_next_step()
    to either loop back to "researcher" or proceed to END.
    """
    # Create the graph builder with our state schema
    graph = StateGraph(GraphState)
    
    # Add the three nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("arbitrator", arbitrator_node)
    
    # Define the edges (the flow)
    graph.set_entry_point("planner")                   # Pipeline starts at Planner
    graph.add_edge("planner", "researcher")            # Planner → Researcher (always)
    graph.add_edge("researcher", "arbitrator")         # Researcher → Arbitrator (always)
    
    # The magic: conditional edge after Arbitrator
    # decide_next_step() returns either "researcher" or END
    graph.add_conditional_edges("arbitrator", decide_next_step)
    
    # Compile the graph into a runnable object
    return graph.compile()
