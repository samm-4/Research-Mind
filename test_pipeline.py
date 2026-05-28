import os
import re
import sys

# Prevent UnicodeEncodeError on Windows terminals
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import core.config  # This triggers the .env loading before anything else runs
from core.graph import build_research_graph


def main():
    user_query = "Solid-state lithium batteries silicon anode vs lithium metal anode volume expansion"
    print("=" * 50)
    print("       ResearchMind — Test Run")
    print(f"Topic: {user_query}")
    print("=" * 50)

    # Build the LangGraph state machine
    graph = build_research_graph()

    # Run the full pipeline
    final_state = graph.invoke({
        "user_query": user_query,
        "subqueries": [],
        "current_subquery_index": 0,
        "current_retry_count": 0,
        "current_researcher_outputs": [],
        "arbitrator_verdicts": [],
        "final_results": []
    })

    print("\n" + "=" * 50)
    print("       FINAL SYNTHESIZED REPORT")
    print("=" * 50)

    report = final_state.get("synthesized_report", "")
    if report:
        print(report)
        os.makedirs("reports", exist_ok=True)
        safe_query = re.sub(r'[^a-zA-Z0-9_\-]+', '_', user_query).strip('_')[:50]
        filename = f"reports/report_{safe_query}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[Exporter] Final report saved to: {filename}")
    else:
        print("\n  No synthesized report was generated.")


if __name__ == "__main__":
    main()
