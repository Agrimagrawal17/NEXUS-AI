from fastapi import FastAPI, UploadFile, File
from app.services.data_profiler import profile_dataset
import tempfile
import os

app = FastAPI(
    title="NEXUS AI",
    description="Autonomous Machine Learning & AI Platform",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "NEXUS AI is running"
    }


@app.post("/analyze")
async def analyze_dataset(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        profile = profile_dataset(temp_file_path)

        return {
            "filename": file.filename,
            "profile": profile
        }

    finally:
        os.remove(temp_file_path)