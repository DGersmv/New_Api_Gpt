from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/chat")
async def proxy_chat(request: Request):
    body = await request.json()
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers)
        return response.json()
