# ResearchMind

A multi-agent research pipeline.
Currently implements Planner and Researcher_1 with RAG.

## Local Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python -m venv myenv
   ```

3. **Activate the virtual environment:**
   - Windows: `myenv\Scripts\activate`
   - Mac/Linux: `source myenv/bin/activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Environment Variables:**
   Copy `.env.example` to `.env` and fill in your API keys. The `.env` file is excluded from version control for security.
   *(Note: You will need GROQ_API_KEY and GEMINI_API_KEY)*

6. **Run the project:**
   ```bash
   python main.py
   ```
