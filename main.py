from fastapi import FastAPI, Request
import openai
import os

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🧠 Временное хранилище сессий (в RAM)
session_threads = {}

ASSISTANT_ID = "asst_rm9rPlWkpOXf3f1EuUJ7AVzt"

@app.get("/")
def root():
    return {"status": "ok", "message": "Assistant with memory is ready."}


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    user_input = data.get("message")

    if not session_id or not user_input:
        return {"error": "session_id and message are required"}

    # 🧠 Получаем или создаём thread
    thread_id = session_threads.get(session_id)
    if not thread_id:
        thread = openai.beta.threads.create()
        thread_id = thread.id
        session_threads[session_id] = thread_id

    # 📩 Отправляем сообщение от пользователя
    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_input
    )

    # ▶️ Запускаем выполнение ассистента
    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )

    # ⏳ Ждём завершения run
    while True:
        run = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run.status in ["completed", "failed", "cancelled", "expired"]:
            break

    if run.status != "completed":
        return {"error": f"Run failed with status: {run.status}"}

    # ✅ Получаем последний ответ
    messages = openai.beta.threads.messages.list(thread_id=thread_id, order="desc")
    for msg in messages.data:
        if msg.role == "assistant":
            return {"answer": msg.content[0].text.value}

    return {"error": "No assistant reply found."}
