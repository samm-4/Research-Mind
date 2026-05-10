"""
main.py
Entry point for ResearchMind.

This file is intentionally simple — it just:
  1. Loads environment variables (.env)
  2. Builds the LangGraph pipeline
  3. Kicks it off with the user's query
  4. Prints the final accepted results

All the orchestration logic (looping, retries, routing) lives
inside core/graph.py. main.py is just the "start button".
"""
import core.config  # This triggers the .env loading before anything else runs
from core.graph import build_research_graph


def main():
    print("=" * 50)
    print("       ResearchMind — Multi-Agent Pipeline")
    print("=" * 50)
    user_query = input("\nEnter a research topic: ")

    # Build the LangGraph state machine
    graph = build_research_graph()

    # Run the full pipeline by providing the initial state
    # LangGraph handles: Planner → Researcher → Arbitrator → (retry?) → ...
    final_state = graph.invoke({
        "user_query": user_query,
        "subqueries": [],
        "current_subquery_index": 0,
        "current_retry_count": 0,
        "researcher_outputs": [],
        "arbitrator_verdicts": [],
        "final_results": []
    })

    # ── Print the final accepted results ──
    print("\n" + "=" * 50)
    print("       FINAL RESEARCH RESULTS")
    print("=" * 50)

    for i, result in enumerate(final_state["final_results"], 1):
        print(f"\n{'─' * 40}")
        print(f"  Subquery {i}: {result.subquery}")
        print(f"{'─' * 40}")
        print(f"  Summary:\n    {result.summary}")
        print(f"\n  Sources: {', '.join(result.sources) if result.sources else 'None'}")
        print(f"  Confidence: {result.confidence}")

    if not final_state["final_results"]:
        print("\n  No results were accepted by the Arbitrator.")


if __name__ == "__main__":
    main()
