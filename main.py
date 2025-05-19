from fastapi import FastAPI, Request, UploadFile, File
import httpx
import os
import tempfile
import shutil
import zipfile
import openai

app = FastAPI()

# 🔑 Подключение к OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Корень (для проверки в браузере)
@app.get("/")
def root():
    return {"status": "ok", "message": "OpenAI Archicad Proxy is running."}

# ✅ Чат с ассистентом через /chat
@app.post("/chat")
async def proxy_chat(request: Request):
    body = await request.json()
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=body,
            headers=headers
        )
        return response.json()

# ✅ Загрузка ZIP архива с SDK и отправка нужных файлов в OpenAI
@app.post("/upload-archicad-examples")
async def upload_examples(file: UploadFile = File(...)):
    allowed_ext = (".cpp", ".c", ".h", ".hpp", ".grc", ".rc2", ".xml", ".txt", ".module")
    uploaded_ids = []

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "sdk.zip")
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(tmpdir)

        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if fname.lower().endswith(allowed_ext):
                    full_path = os.path.join(root, fname)
                    try:
                        with open(full_path, "rb") as f:
                            result = openai.files.create(file=f, purpose="assistants")
                            print(f"✅ Загружен: {fname} → {result.id}")
                            uploaded_ids.append(result.id)
                    except Exception as e:
                        print(f"❌ Ошибка: {fname} — {e}")

    return {
        "status": "ok",
        "uploaded_file_ids": uploaded_ids,
        "count": len(uploaded_ids)
    }
