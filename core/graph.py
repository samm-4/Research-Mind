"""
graph.py
LangGraph state machine for ResearchMind.

Orchestrates the pipeline flow:
  Planner ➔ Researcher ➔ Arbitrator (loops for each subquery) ➔ Combined Synthesizer ➔ END
"""
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END

from agents.planner import generate_subqueries
from agents.researcher import research_subquery_1, research_subquery_2, research_subquery_3
from agents.arbitrator import evaluate_outputs
from agents.synthesizer import synthesize_all_results
from core.schemas import ResearcherOutput, ArbitratorVerdict


# ── Configuration ──
MAX_RETRIES = 2  # Max retry attempts per subquery before force-accepting


# ──────────────────────────────────────────────
#  GRAPH STATE
# ──────────────────────────────────────────────
class GraphState(TypedDict):
    user_query: str                                    # The original user question
    subqueries: List[str]                              # 3 subqueries from the Planner
    requires_quantitative_metrics: bool                 # Whether the research requires strict physical/chemical metrics
    current_subquery_index: int                        # Which subquery we're currently on
    current_retry_count: int                           # How many retries for current subquery
    current_researcher_outputs: List[ResearcherOutput] # Outputs from all 3 researchers for CURRENT subquery
    arbitrator_verdicts: List[ArbitratorVerdict]        # Accumulated verdicts from Arbitrator
    final_results: List[ResearcherOutput]              # Accepted researcher outputs (one per subquery)
    synthesized_report: str                            # Final unified markdown report
    progress_queue: Any                                # Optional queue for real-time progress updates


# ──────────────────────────────────────────────
#  PROGRESS STREAMING HELPER
# ──────────────────────────────────────────────
def publish_progress(state: GraphState, event_type: str, message: str, data: dict = None):
    """
    Pushes progress updates to a progress_queue if it exists in the state.
    """
    queue = state.get("progress_queue")
    if queue is not None:
        try:
            queue.put({
                "event": event_type,
                "message": message,
                "data": data or {}
            })
        except Exception:
            pass


# ──────────────────────────────────────────────
#  NODE 1: PLANNER
# ──────────────────────────────────────────────
def planner_node(state: GraphState) -> dict:
    """
    Takes the user query and generates 3 subqueries.
    This node runs exactly once at the start of the pipeline.
    """
    publish_progress(state, "planner_start", "Analyzing search topic and planning research strategy...")
    print("\n[Planner] Thinking...")
    planner_result = generate_subqueries(state["user_query"])
    subqueries = planner_result.get("subqueries", [])
    req_quant = planner_result.get("requires_quantitative_metrics", False)

    print(f"[Planner] Generated exactly 3 subqueries (Requires Quantitative: {req_quant}):")
    for i, sq in enumerate(subqueries, 1):
        print(f"  {i}. {sq}")

    publish_progress(state, "planner_end", f"Planner generated 3 subqueries (Requires Quantitative: {req_quant}).", {
        "subqueries": subqueries,
        "requires_quantitative_metrics": req_quant
    })

    return {
        "subqueries": subqueries,
        "requires_quantitative_metrics": req_quant,
        "current_subquery_index": 0,
        "current_retry_count": 0,
        "current_researcher_outputs": [],
        "arbitrator_verdicts": [],
        "final_results": [],
        "synthesized_report": ""
    }


# ──────────────────────────────────────────────
#  NODE 2: RESEARCHERS
# ──────────────────────────────────────────────
def researcher_node(state: GraphState) -> dict:
    """
    Runs ALL 3 researcher agents sequentially on the CURRENT subquery.
    """
    idx = state["current_subquery_index"]
    subquery = state["subqueries"][idx]
    retry = state["current_retry_count"]
    req_quant = state.get("requires_quantitative_metrics", False)

    # Add a small pacing delay to naturally separate API requests and avoid rate limits
    import time
    if idx > 0 or retry > 0:
        print(f"\n  [Pacing] Pausing for 3.0s to avoid API rate limits...")
        publish_progress(state, "pacing_start", "Pacing: pausing 3 seconds to stay under rate limits...")
        time.sleep(3.0)

    publish_progress(state, "researcher_start", f"Running researchers for Subquery {idx + 1}/{len(state['subqueries'])}...", {
        "subquery": subquery,
        "index": idx,
        "retry": retry
    })

    print(f"\n{'='*50}")
    print(f"  SUBQUERY {idx + 1} (attempt {retry + 1}): {subquery}")
    print(f"{'='*50}")

    # Run all 3 researchers on the same subquery
    print(f"\n  --- Researcher 1 (Gemini / Balanced) ---")
    publish_progress(state, "researcher_progress", "Researcher 1 (Gemini / Balanced) is searching and analyzing...")
    result_1 = research_subquery_1(subquery, requires_quantitative=req_quant)
    publish_progress(state, "researcher_progress", f"Researcher 1 finished (Confidence: {result_1.confidence:.2f})", {"sources": result_1.sources})

    print(f"\n  --- Researcher 2 (Groq / Analytical) ---")
    publish_progress(state, "researcher_progress", "Researcher 2 (Groq / Llama-3.3) is searching and analyzing...")
    result_2 = research_subquery_2(subquery, requires_quantitative=req_quant)
    publish_progress(state, "researcher_progress", f"Researcher 2 finished (Confidence: {result_2.confidence:.2f})", {"sources": result_2.sources})

    print(f"\n  --- Researcher 3 (Groq / Llama-8B / Critical) ---")
    publish_progress(state, "researcher_progress", "Researcher 3 (Groq / Llama-8B) is deep searching and reviewing...")
    result_3 = research_subquery_3(subquery, requires_quantitative=req_quant)
    publish_progress(state, "researcher_progress", f"Researcher 3 finished (Confidence: {result_3.confidence:.2f})", {"sources": result_3.sources})

    # Combine all sources found in this subquery
    combined_sources = list(set(result_1.sources + result_2.sources + result_3.sources))
    publish_progress(state, "researcher_end", f"All researchers completed for Subquery {idx + 1}.", {"sources": combined_sources})

    return {
        "current_researcher_outputs": [result_1, result_2, result_3]
    }


# ──────────────────────────────────────────────
#  NODE 3: ARBITRATOR
# ──────────────────────────────────────────────
def arbitrator_node(state: GraphState) -> dict:
    """
    Evaluates ALL 3 ResearcherOutputs for the current subquery.
    If accepted (or max retries reached), appends the best output to final_results
    and advances the subquery index.
    """
    idx = state["current_subquery_index"]
    subquery = state["subqueries"][idx]
    outputs = state["current_researcher_outputs"]
    retries = state["current_retry_count"]
    req_quant = state.get("requires_quantitative_metrics", False)

    publish_progress(state, "arbitrator_start", f"Arbitrator evaluating outputs for Subquery {idx + 1}...")

    print(f"\n[Arbitrator] Evaluating {len(outputs)} researcher outputs for: '{subquery}'")
    verdict = evaluate_outputs(subquery, outputs, requires_quantitative=req_quant)

    # Check if we accept this subquery's findings
    if verdict.accepted or retries >= MAX_RETRIES:
        if verdict.accepted:
            print(f"  ✅ ACCEPTED — {verdict.reason}")
            publish_progress(state, "arbitrator_end", f"ACCEPTED — {verdict.reason}", {
                "accepted": True,
                "reason": verdict.reason,
                "confidence": verdict.best_output.confidence if verdict.best_output else 0.0
            })
        else:
            print(f"  ⚠️  Max retries reached. Force-accepting best available results.")
            publish_progress(state, "arbitrator_end", "Max retries reached. Force-accepting best available results.", {
                "accepted": True,
                "reason": "Max retries reached",
                "confidence": verdict.best_output.confidence if verdict.best_output else 0.0
            })
            
        best_out = verdict.best_output if verdict.best_output else outputs[0]

        return {
            "arbitrator_verdicts": state["arbitrator_verdicts"] + [verdict],
            "final_results": state["final_results"] + [best_out],
            "current_subquery_index": idx + 1,
            "current_retry_count": 0,
            "current_researcher_outputs": []  # Clear for the next subquery
        }
    else:
        print(f"  ❌ REJECTED — {verdict.reason}")
        publish_progress(state, "arbitrator_end", f"REJECTED — {verdict.reason}", {
            "accepted": False,
            "reason": verdict.reason
        })
        return {
            "arbitrator_verdicts": state["arbitrator_verdicts"] + [verdict],
            "current_retry_count": retries + 1,
            "current_researcher_outputs": []  # Clear for retry
        }


# ──────────────────────────────────────────────
#  NODE 4: COMBINED SYNTHESIZER
# ──────────────────────────────────────────────
def synthesizer_node(state: GraphState) -> dict:
    """
    Runs ONCE at the end of the entire pipeline.
    Synthesizes the accepted researcher results for all 3 subqueries into a single dossier.
    """
    user_query = state["user_query"]
    results = state["final_results"]

    publish_progress(state, "synthesizer_start", "Compiling final master dossier and bibliography...")

    synthesized_report = synthesize_all_results(user_query, results)

    publish_progress(state, "synthesizer_end", "Master research dossier successfully generated!")

    return {
        "synthesized_report": synthesized_report
    }


# ──────────────────────────────────────────────
#  CONDITIONAL ROUTING EDGES
# ──────────────────────────────────────────────
def decide_next_step(state: GraphState) -> str:
    """
    Routing function called by LangGraph after the Arbitrator node.
    """
    idx = state["current_subquery_index"]
    total_subqueries = len(state["subqueries"])
    retries = state["current_retry_count"]

    # If retries is 0, it means we just advanced the subquery index
    # (either because the result was accepted or we hit max retries)
    if retries == 0:
        if idx >= total_subqueries:
            # All subqueries have been researched. Go to synthesizer node
            return "synthesizer"
        else:
            # Move on to research the next subquery
            return "researcher"
    else:
        # Retry the current subquery index
        return "researcher"


# ──────────────────────────────────────────────
#  BUILD THE GRAPH
# ──────────────────────────────────────────────
def build_research_graph():
    """
    Constructs and compiles the LangGraph state machine.
    """
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("arbitrator", arbitrator_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Define edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "arbitrator")

    # Conditional edge: loops researcher or proceeds to synthesizer at the end
    graph.add_conditional_edges("arbitrator", decide_next_step)
    
    # Synthesizer runs once and terminates the graph
    graph.add_edge("synthesizer", END)

    return graph.compile()
