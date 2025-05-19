from fastapi import FastAPI, Request, UploadFile, File, Form
import openai
import os
import asyncio

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🧠 Сохраняем соответствие session_id → thread_id
session_threads = {}

# 🔗 Основной ассистент
DEFAULT_ASSISTANT_ID = "asst_rm9rPlWkpOXf3f1EuUJ7AVzt"

@app.get("/")
def root():
    return {"status": "ok", "message": "Archicad Assistant server is running."}


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    user_input = data.get("message")

    if not session_id or not user_input:
        return {"error": "session_id and message are required"}

    # 🔁 Создаём thread при первом обращении
    thread_id = session_threads.get(session_id)
    if not thread_id:
        thread = openai.beta.threads.create()
        thread_id = thread.id
        session_threads[session_id] = thread_id

    # ➕ Отправляем сообщение в thread
    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_input
    )

    # ▶️ Запускаем ассистента
    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=DEFAULT_ASSISTANT_ID
    )

    # ⏳ Ожидаем завершения run
    while True:
        run = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run.status in ["completed", "failed", "cancelled", "expired"]:
            break
        await asyncio.sleep(1)

    if run.status != "completed":
        return {"error": f"Run failed: {run.status}"}

    # 📤 Получаем ответ
    messages = openai.beta.threads.messages.list(thread_id=thread_id, order="desc")
    for msg in messages.data:
        if msg.role == "assistant":
            return {"answer": msg.content[0].text.value}

    return {"error": "No assistant response found."}


@app.post("/upload-to-vectorstore")
async def upload_to_vectorstore(
    file: UploadFile = File(...),
    assistant_id: str = Form(...),
    vector_store_id: str = Form(...)
):
    try:
        # 1. Загружаем файл в OpenAI
        uploaded = openai.files.create(
            file=file.file,
            purpose="assistants"
        )

        # 2. Добавляем в векторстор
        openai.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=[uploaded.id]
        )

        # 3. Обновляем ассистента
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
