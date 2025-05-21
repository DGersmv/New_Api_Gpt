from fastapi import FastAPI, Request, UploadFile, File, Form, Body
import openai
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime
import uuid

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")
DEFAULT_ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# === Постоянное хранение session_id → thread_id и мета
SESSIONS_FILE = Path("sessions.json")
session_threads = {}

if SESSIONS_FILE.exists():
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            session_threads = json.load(f)
    except Exception as e:
        print(f"⚠️ Не удалось загрузить sessions.json: {e}")

def save_sessions():
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(session_threads, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Не удалось сохранить sessions.json: {e}")

@app.get("/list-sessions")
def list_sessions():
    return session_threads

@app.post("/create-session")
def create_session(data: dict = Body(...)):
    title = data.get("title", "Новый проект")
    session_id = data.get("session_id") or str(uuid.uuid4())
    assistant_id = data.get("assistant_id") or DEFAULT_ASSISTANT_ID

    try:
        thread = openai.beta.threads.create()
    except Exception as e:
        return {"error": f"Ошибка при создании thread: {str(e)}"}

    session_threads[session_id] = {
        "thread_id": thread.id,
        "created": datetime.now().isoformat(),
        "last_used": datetime.now().isoformat(),
        "title": title,
        "assistant_id": assistant_id
    }
    save_sessions()
    return {"session_id": session_id, "thread_id": thread.id, "title": title}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    message = data.get("message")
    assistant_id = data.get("assistant_id") or DEFAULT_ASSISTANT_ID

    if not session_id or not message:
        return {"error": "session_id и message обязательны"}

    session = session_threads.get(session_id)
    if not session:
        return {"error": f"Сессия '{session_id}' не найдена"}

    thread_id = session["thread_id"]
    session["last_used"] = datetime.now().isoformat()
    save_sessions()

    try:
        openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )

        run = openai.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
    except Exception as e:
        return {"error": f"Ошибка запуска ассистента: {str(e)}"}

    while True:
        run = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run.status in ["completed", "failed", "cancelled", "expired"]:
            break
        await asyncio.sleep(1)

    if run.status != "completed":
        error_message = getattr(run, "last_error", {}).get("message", "unknown reason")
        return {"error": f"Run failed: {run.status} - {error_message}"}

    try:
        messages = openai.beta.threads.messages.list(thread_id=thread_id, order="desc")
    except Exception as e:
        return {"error": f"Не удалось получить ответ: {str(e)}"}

    for msg in messages.data:
        if msg.role == "assistant":
            return {"answer": msg.content[0].text.value}

    return {"error": "Ассистент не дал ответа."}

@app.post("/upload-to-vectorstore")
async def upload_to_vectorstore(
    file: UploadFile = File(...),
    assistant_id: str = Form(...),
    vector_store_id: str = Form(...)
):
    try:
        uploaded = openai.files.create(file=file.file, purpose="assistants")
        openai.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=[uploaded.id]
        )
        openai.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )
        return {"status": "success", "file_id": uploaded.id}
    except Exception as e:
        return {"error": str(e)}
