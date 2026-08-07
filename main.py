from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from app.zaira import zaira_response_stream

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent


class Query(BaseModel):
    text: str


@app.post("/chat")
def chat(q: Query):
    return StreamingResponse(zaira_response_stream(q.text), media_type="text/plain")


app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")
