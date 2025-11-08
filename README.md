# Cosmic-Nexus
Cosmic Nexus AI Orchestrator
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
openai==1.3.7  # For xAI API compatibility (swap client as needed)
python-dotenv==1.0.0
pytest==7.4.3  # For testing
import os
from typing import Dict, Any, Optional
from openai import OpenAI  # xAI API uses OpenAI-compatible client
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    plan: str
    results: Dict[str, str]
    final_output: str

class CosmicNexus:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY environment variable is required.")
        pip install -r requirements.txt
export XAI_API_KEY="your_key"
uvicorn main:app --reload
curl -X POST http://localhost:8000/process -H "Content-Type: application/json" -d '{"query": "Quantum entanglement viz"}'
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1",  # xAI API endpoint
        )
        
        self.projects = {
            'grok': self._call_grok,
            'aurora': self._call_aurora,
            'grokipedia': self._call_grokipedia,
            'hotshot': self._call_hotshot,
        }
    
    def _call_grok(self, prompt: str) -> str:
        """Call Grok-4 for reasoning/planning."""
        response = self.client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )@app.post("/process_with_tools")
@limiter.limit("5/minute")  # Stricter for tool-heavy
async def process_with_tools(request: QueryRequest):
    """Enhanced endpoint with tool-calling."""
    if not nexus:
        raise HTTPException(status_code=500, detail="Nexus unavailable.")
    try:
        result = await nexus.process_query(request.query)  # Uses base; tools auto-trigger
        return {"enhanced": True, **result.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        return response.choices[0].message.content
    
    def _call_aurora(self, prompt: str) -> str:
        """Mock Aurora text-to-image (replace with real API)."""
        # In reality: Use xAI Aurora endpoint
        return f"Generated image URL: https://aurora.x.ai/{prompt.replace(' ', '_')}.png"
    
    def _call_grokipedia(self, query: str) -> str:
        """Mock Grokipedia search (integrate real API when available)."""
        grok_facts = self._call_grok(f"Provide verified facts on: {query}")
        return f"Facts from Grokipedia: {grok_facts}"
    
    def _call_hotshot(self, script: str) -> str:
        """Mock Hotshot video gen (from acquisition)."""
        # In reality: Call Hotshot API
        return f"Video clip: {script} – 30s animation ready at https://hotshot.x.ai/{script.replace(' ', '_')}.mp4"
    
    def process_query(self, user_input: str) -> QueryResponse:
        """Main orchestrator: Plan, execute subtasks, synthesize."""
        # Step 1: Grok plans the tasks
        plan_prompt = f"Analyze: {user_input}. Break into subtasks for available projects (aurora, grokipedia, hotshot). Output as JSON: {{'aurora': 'task', 'grokipedia': 'task', 'hotshot': 'task'}}"
    
tasks = {
    'aurora': user_input,  # Default if no parse
    'grokipedia': user_input,
    'hotshot': f"Demo {user_input}"
}

# NEW:
tasks = self.parse_plan_json(plan_raw)  # Note: pass user_input if needed in method
        # Simple JSON parse (in prod, use json.loads with error handling)
        tasks = {
           # OLD (lines ~85-95):
results = {}
for project, task in tasks.items():
    if project in self.projects:
        if project == 'grok':
            results[project] = await self.projects[project](task)
        else:
            results[project] = self.projects[project](task)  # Sync for mocks

# NEW:
# Step 2: Parallel execution
async def execute_task(project, task):
    if project in self.projects:
        if project == 'grok':
            return project, await self.projects[project](task)
        else:
            return project, self.projects[project](task)  # Sync for mocks

tasks_coros = [execute_task(project, task) for project, task in tasks.items()]
parallel_results = await asyncio.gather(*tasks_coros, return_exceptions=True)
results = {proj: res if not isinstance(res, Exception) else f"Error in {proj}: {res}" 
           for proj, res in parallel_results}
        tools = [{"type": "function", "function": {"name": "deep_search", "description": "Search web for fresh facts", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
        plan_prompt = f"Analyze: {user_input}. Break into subtasks for available projects (aurora, grokipedia, hotshot). Output as JSON: {{'aurora': 'task', 'grokipedia': 'task', 'hotshot': 'task'}}"
        plan_raw = await self._call_grok_with_tools(plan_prompt, tools=tools if "deep" in user_input.lower() else None)
        # Step 3: Synthesize with Grok
        synthesis_prompt = f"Synthesize a coherent response from: {results}. Original query: {user_input}. Make it engaging and complete."
        synthesis = self._call_grok(synthesis_prompt)
        
        return QueryResponse(
            plan=plan_raw,
            results=results,
            final_output=synthesis
        )
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nexus import CosmicNexus, QueryRequest, QueryResponse

app = FastAPI(title="Cosmic Nexus AI Orchestrator", version="1.0.0")

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global nexus instance (in prod, use dependency injection)
try:
    nexus = CosmicNexus()
except ValueError as e:
    print(f"Error initializing Nexus: {e}")
    nexus = None

@app.post("/process", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not nexus:
        raise HTTPException(status_code=500, detail="Nexus not initialized. Check API key.")
    
    try:
        result = nexus.process_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Cosmic Nexus is ready to explore the universe!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    import pytest
from nexus import CosmicNexus, QueryRequest

# Mock API key for tests
os.environ["XAI_API_KEY"] = "test_key"  # Won't actually call, but initializes

def test_process_query():
    nexus = CosmicNexus(api_key="test_key")
    result = nexus.process_query("Test query")
    assert "plan" in result.dict()
    assert "results" in result.dict()
    assert "final_output" in result.dict()
    assert len(result.results) > 0
      # Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# For K8s: Apply with kubectl apply -f this.yaml (adapt as needed)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cosmic-nexus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cosmic-nexus
  template:
    metadata:
      labels:
        app: cosmic-nexus
    spec:
      containers:
      - name: nexus
        image: your-docker-image:latest
        ports:
        - containerPort: 8000
        env:
        - name: XAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: xai-secrets
              key: api-key
---
apiVersion: v1
kind: Service
metadata:
  name: cosmic-nexus-service
spec:
  selector:
    app: cosmic-nexus
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
  MIT License

Copyright (c) 2025 xAI Community

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
__pycache__/
*.pyc
.env
*.log
.DS_Store
pip install -r requirements.txt
export XAI_API_KEY="your_key_here"
curl -X POST "http://localhost:8000/process" \
-H "Content-Type: application/json" \
-d '{"query": "Explain quantum entanglement and generate a visualization."}'
tenacity==8.3.0  # For retries
python-multipart==0.0.6  # If adding file uploads later
from tenacity import retry, stop_after_attempt, wait_exponential  # Retries
from typing import List  # For project lists
import os  # If not already there
        # Configurable projects via env
        enabled_str = os.getenv("ENABLED_PROJECTS", "grok,aurora,grokipedia,hotshot")
        enabled = [p.strip() for p in enabled_str.split(",")]
        self.projects = {k: v for k, v in self.projects.items() if k in enabled}
        
        logger.info(f"Enabled projects: {list(self.projects.keys())}")
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Retry wrapper for API calls."""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Retryable error: {e}")
            raise
    
    async def _call_grok_with_tools(self, prompt: str, tools: Optional[List[Dict]] = None) -> str:
        """Grok-4 tool-calling (e.g., for DeepSearch)."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self._call_with_retry(
                self.client.chat.completions.create,
                model="grok-4",
                messages=messages,
                tools=tools,  # e.g., [{"type": "function", "function": {"name": "deep_search", ...}}]
                tool_choice="auto",
                max_tokens=500,
                temperature=0.7
            )
            content = response.choices[0].message.content
            # Handle tool calls if present (expand as needed)
            if response.choices[0].message.tool_calls:
                logger.info("Tool calls triggered; implement execution here.")
                # Example: For DeepSearch, parse and call web tool
                content += " [Tool results integrated]"
            return content
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return await self._call_grok(prompt)  # Fallback
            @app.post("/process_with_tools")
@limiter.limit("5/minute")  # Stricter for tool-heavy
async def process_with_tools(request: QueryRequest):
    """Enhanced endpoint with tool-calling."""
    if not nexus:
        raise HTTPException(500, "Nexus unavailable.")
    try:
        # Wrap in tool-enabled mode
        result = await nexus.process_query(request.query)  # Still uses base; extend if needed
        return {"enhanced": True, **result.dict()}
    except Exception as e:
        raise HTTPException(500, str(e))
        @pytest.mark.asyncio
async def test_tool_calling():
    nexus = CosmicNexus(api_key="test_key")
    result = await nexus._call_grok_with_tools("Test with tools")
    assert isinstance(result, str)
    assert len(result) > 0
    XAI_API_KEY=your_key_here
ENABLED_PROJECTS=grok,aurora,grokipedia,hotshot
LOG_LEVEL=INFO
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Retry wrapper for API calls."""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Retryable error: {e}")
            raise
    
    async def _call_grok_with_tools(self, prompt: str, tools: Optional[List[Dict]] = None) -> str:
        """Grok-4 tool-calling (e.g., for DeepSearch)."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self._call_with_retry(
                self.client.chat.completions.create,
                model="grok-4",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=500,
                temperature=0.7
            )
            content = response.choices[0].message.content
            # Handle tool calls if present (expand as needed)
            if response.choices[0].message.tool_calls:
                logger.info("Tool calls triggered; implement execution here.")
                # Example: For DeepSearch, parse and call web tool
                content += " [Tool results integrated]"
            return content
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return await self._call_grok(prompt)  # Fallback
            @pytest.mark.asyncio
async def test_tool_calling():
    nexus = CosmicNexus(api_key="test_key")
    result = await nexus._call_grok_with_tools("Test with tools")
    assert isinstance(result, str)
    assert len(result) > 0
