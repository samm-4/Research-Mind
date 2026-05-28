import os
import re
import sys
import queue
import asyncio
import threading
from typing import List

# Prevent UnicodeEncodeError on Windows terminals
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Add the parent directory (root of the workspace) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import core.config  # Load environment variables (.env) before anything else

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.graph import build_research_graph

app = FastAPI(title="ResearchMind Interface")

# Mount static files folder
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
def read_root():
    """Serves the main single-page web app."""
    return FileResponse("frontend/static/index.html")


class LoadReportRequest(BaseModel):
    filename: str


@app.get("/api/history")
def get_history():
    """Returns a list of previously generated reports from the reports/ folder."""
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        return []
    
    files = []
    try:
        for f in os.listdir(reports_dir):
            if f.endswith(".md") and f.startswith("report_"):
                path = os.path.join(reports_dir, f)
                stat = os.stat(path)
                # Form a reader-friendly title from the filename
                title = f.replace("report_", "").replace(".md", "").replace("_", " ")
                files.append({
                    "filename": f,
                    "title": title,
                    "sizeBytes": stat.st_size,
                    "createdTime": stat.st_mtime
                })
        # Sort by creation time descending (newest first)
        files.sort(key=lambda x: x["createdTime"], reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return files


@app.get("/api/history/load")
def load_report(filename: str):
    """Loads and returns the content of a specific report."""
    # Prevent directory traversal attacks
    filename = os.path.basename(filename)
    path = os.path.join("reports", filename)
    
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"filename": filename, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/research")
async def websocket_research(websocket: WebSocket):
    """Handles real-time research query streaming over WebSockets."""
    await websocket.accept()
    
    try:
        # Wait for the client to send the query
        data = await websocket.receive_json()
        user_query = data.get("query", "").strip()
        
        if not user_query:
            await websocket.send_json({"event": "error", "message": "Query cannot be empty"})
            await websocket.close()
            return
            
        progress_queue = queue.Queue()
        result_holder = {}
        
        # Build graph
        graph = build_research_graph()
        
        # Target thread function to run the LangGraph pipeline
        def run_pipeline():
            try:
                final_state = graph.invoke({
                    "user_query": user_query,
                    "subqueries": [],
                    "current_subquery_index": 0,
                    "current_retry_count": 0,
                    "current_researcher_outputs": [],
                    "arbitrator_verdicts": [],
                    "final_results": [],
                    "progress_queue": progress_queue
                })
                report = final_state.get("synthesized_report", "")
                result_holder["synthesized_report"] = report
                if report:
                    os.makedirs("reports", exist_ok=True)
                    safe_query = re.sub(r'[^a-zA-Z0-9_\-]+', '_', user_query).strip('_')[:50]
                    filename = f"reports/report_{safe_query}.md"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(report)
            except Exception as e:
                # Catch rate-limits and other errors
                progress_queue.put({"event": "error", "message": str(e)})
            finally:
                progress_queue.put({"event": "complete", "message": "done"})
                
        # Start the pipeline thread
        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()
        
        # Stream events from progress_queue to the client via WebSocket
        while True:
            # Poll the queue in a non-blocking way using an async thread helper
            item = await asyncio.to_thread(progress_queue.get)
            
            if item["event"] == "complete":
                # Pipeline finished
                report = result_holder.get("synthesized_report", "")
                await websocket.send_json({
                    "event": "complete",
                    "message": "Research complete!",
                    "report": report
                })
                break
            elif item["event"] == "error":
                # Pipeline failed
                await websocket.send_json({
                    "event": "error",
                    "message": item["message"]
                })
                break
            else:
                # Standard progress event, pass it to client
                await websocket.send_json(item)
                
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    
    def open_browser():
        import time
        time.sleep(1.5)
        print("\n[Browser] Opening http://127.0.0.1:8000 in your default browser...")
        webbrowser.open("http://127.0.0.1:8000")

    # Start browser opener thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start the server on port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
