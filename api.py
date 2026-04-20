from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import os
import tempfile
import shutil
import json

from AlignPipeline import script, LANGUAGES, OUTPUT_DIR
from lingua import Language

app = FastAPI(title="Transcript Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/align")
async def align_audio(
    audio: UploadFile = File(...),
    transcript: str = Form(...),
    download_models: bool = Form(False),
    output_format: str = Form("json"),
):
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, audio.filename)
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        try:
            segments = json.loads(transcript)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON transcript")

        try:
            result = script(
                audio_path=audio_path,
                transcript=segments,
                temp_dir=temp_dir,
                languages=LANGUAGES,
                download_models=download_models,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        base_name = os.path.basename(audio_path).split(".")[0]

        if output_format == "json":
            output_path = os.path.join(temp_dir, base_name + "_Aligned.json")
            return FileResponse(
                output_path,
                media_type="application/json",
                filename=f"{base_name}_Aligned.json",
            )
        elif output_format == "csv":
            output_path = os.path.join(temp_dir, base_name + "_Aligned.csv")
            return FileResponse(
                output_path, media_type="text/csv", filename=f"{base_name}_Aligned.csv"
            )
        elif output_format == "textgrid":
            output_path = os.path.join(temp_dir, base_name + "_Aligned.TextGrid")
            return FileResponse(
                output_path,
                media_type="application/octet-stream",
                filename=f"{base_name}_Aligned.TextGrid",
            )
        else:
            return {"segments": result}


@app.get("/health")
async def health():
    return {"status": "healthy"}
