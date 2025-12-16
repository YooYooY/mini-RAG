from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import run_agent

app = FastAPI()


class AgentRequest(BaseModel):
    query: str


class AgentResponse(BaseModel):
    answer: str
    status: str = "ok"


@app.post("/agent/ask", response_model=AgentResponse)
def ask_agent(req: AgentRequest):
    answer = run_agent(req.query)
    return AgentResponse(answer=answer)
