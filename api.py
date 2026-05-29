from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import os
import tempfile
import shutil
import json
import zipfile

import TranscribePipeline as TP
import AlignPipeline as AP
import EditPipeline as EP
import ShignPipeline as SP
import ConfigPipeline as Config
from lingua import Language

app = FastAPI(title="Transcript Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/shign")
async def shift_align(
    file1: UploadFile = File(..., description="First audio/video file"),
    file2: UploadFile = File(..., description="Second audio/video file"),
    pad: bool = Form(False, description="Create new files with padding applied"),
    pad_mode: str = Form("video", description="Which media to pad/cut: 'video' or 'audio'"),
):
    """Upload two audio/video files to align them using Shign.

    Args:
        file1: First audio or video file (.mp4, .mov, .mp3, .wav)
        file2: Second audio or video file (.mp4, .mov, .mp3, .wav)
        pad: If True, creates new files with padding applied to sync them
        pad_mode: Which media type to pad for syncing: 'video' or 'audio'

    Returns:
        JSON with alignment results and delta in milliseconds
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Save uploaded files
        path1 = os.path.join(temp_dir, file1.filename)
        path2 = os.path.join(temp_dir, file2.filename)

        with open(path1, "wb") as f:
            shutil.copyfileobj(file1.file, f)
        with open(path2, "wb") as f:
            shutil.copyfileobj(file2.file, f)

        try:
            output = SP.script(path1, path2, pad=pad, pad_mode=pad_mode)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Handle response based on whether files were aligned
        if pad and output:
            # pad=True: SP.script returns paths to aligned files, create zip
            path1_aligned, path2_aligned = output
            base_name = os.path.basename(path1).rsplit('.', 1)[0]
            zip_filename = f"{base_name}_shigned.zip"
            zip_path = os.path.join(Config.OUTPUT_DIR, zip_filename)

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(path1_aligned, arcname=os.path.basename(path1_aligned))
                zf.write(path2_aligned, arcname=os.path.basename(path2_aligned))

            return FileResponse(zip_path, media_type="application/zip", filename=zip_filename)
        else:
            # pad=False or no alignment needed: return original files info
            return {"status": "aligned", "file1": path1, "file2": path2, "Output": output}

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    method: str = Form(""), # ["faster-whisper", "vosk", ""]
):
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, audio.filename)
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        
        try: 
            result = TP.script(
                audio_path=audio_path,
                temp_dir=temp_dir,
                languages=Config.LANGUAGES,
                method=method,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        return result

@app.post("/json-to-textgrid")
async def convert_json_to_tg(
    file: Optional[UploadFile] = File(None),
    json_text: Optional[str] = Form(None)
):
    """Upload simplified JSON OR paste it, download TextGrid."""
    
    if file:
        content = await file.read()
        data = json.loads(content)
        base_name = file.filename.rsplit('.', 1)[0]
    elif json_text:
        data = json.loads(json_text)
        base_name = "pasted_transcript"
    else:
        raise HTTPException(400, "Provide a .json file or paste the JSON content.")
    
    tg_path = os.path.join(tempfile.gettempdir(), f"{base_name}.TextGrid")
    EP.json_to_textgrid(data, tg_path)
    
    return FileResponse(tg_path, filename=f"{base_name}.TextGrid")

@app.post("/textgrid-to-json")
async def convert_tg_to_json(
    file: UploadFile = File(...),
    tier_index: int = Form(0)
):
    """Upload TextGrid, return JSON data as text for easy copying."""
    if not file.filename.lower().endswith('.textgrid'):
        raise HTTPException(400, "Please upload a .TextGrid file")

    with tempfile.TemporaryDirectory() as temp_dir:
        tg_path = os.path.join(temp_dir, file.filename)
        
        # Save the uploaded file temporarily
        with open(tg_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        try:
            # 1. Convert TextGrid to a Python list/dict
            segments = EP.textgrid_to_json(tg_path, tier_index=tier_index)
            
            # 2. Return segments directly. 
            # FastAPI turns this into a JSON string automatically.
            return segments
            
        except IndexError:
            raise HTTPException(400, detail=f"Tier index {tier_index} not found in TextGrid.")
        except Exception as e:
            raise HTTPException(500, detail=f"Conversion failed: {str(e)}")

@app.post("/m4a-to-wav")
async def convert_m4a_to_wav(
    file: UploadFile = File(..., description="M4A audio file to convert to WAV")
):
    """Upload an M4A file to convert to WAV format.

    The WAV file will be generated with the same filename but with .wav extension.

    Args:
        file: M4A audio file to convert

    Returns:
        The converted WAV file for download
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        m4a_path = os.path.join(temp_dir, file.filename)
        with open(m4a_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            # Convert M4A to WAV
            temp_wav_path = EP.m4a_to_wav(m4a_path)
            wav_filename = os.path.basename(temp_wav_path)
            wav_path = os.path.join(Config.OUTPUT_DIR, wav_filename)
            shutil.move(temp_wav_path, wav_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

        # Return the WAV file for download
        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename=os.path.basename(wav_path)
        )

@app.post("/mp3-to-wav")
async def convert_mp3_to_wav(
    file: UploadFile = File(..., description="MP3 audio file to convert to WAV")
):
    """Upload an MP3 file to convert to WAV format.

    The WAV file will be generated with the same filename but with .wav extension.

    Args:
        file: MP3 audio file to convert

    Returns:
        The converted WAV file for download
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        mp3_path = os.path.join(temp_dir, file.filename)
        with open(mp3_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            # Convert MP3 to WAV
            temp_wav_path = EP.mp3_to_wav(mp3_path)
            wav_filename = os.path.basename(temp_wav_path)
            wav_path = os.path.join(Config.OUTPUT_DIR, wav_filename)
            shutil.move(temp_wav_path, wav_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")

        # Return the WAV file for download
        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename=os.path.basename(wav_path)
        )

@app.post("/mfa")
async def montreal_forced_aligner(
    audio: UploadFile = File(...),
    transcript_file: Optional[UploadFile] = File(None),       # Optional file
    transcript_text: Optional[str] = Form(None),               # Optional text box
    download_models: bool = Form(False),
    languages: Optional[str] = Form("english,spanish,mandarin", description="Comma-separated languages. For example: english,spanish,mandarin")
):
    # Parse comma-separated languages string
    if languages:
        lang_map = {
            "english": Language.ENGLISH,
            "spanish": Language.SPANISH,
            "mandarin": Language.CHINESE,
        }
        selected_languages = [lang_map[lang.strip().lower()] for lang in languages.split(",") if lang.strip().lower() in lang_map]
    else:
        selected_languages = Config.LANGUAGES

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, audio.filename)
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        # Determine which source to use
        if transcript_file:
            content = await transcript_file.read()
            segments = json.loads(content)
        elif transcript_text:
            segments = json.loads(transcript_text)
        else:
            raise HTTPException(status_code=400, detail="Please provide either a JSON file or paste the JSON text.")

        try:
            result = AP.script(
                audio_path=audio_path,
                transcript=segments,
                temp_dir=temp_dir,
                languages=selected_languages,
                download_models=download_models,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        base_name = os.path.basename(audio_path).split(".")[0]

        output_files = {
            f"{base_name}_Aligned.json":      (os.path.join(Config.OUTPUT_DIR, f"{base_name}_Aligned.json"),      "application/json"),
            f"{base_name}_Aligned.csv":       (os.path.join(Config.OUTPUT_DIR, f"{base_name}_Aligned.csv"),       "text/csv"),
            f"{base_name}_Aligned.TextGrid":  (os.path.join(Config.OUTPUT_DIR, f"{base_name}_Aligned.TextGrid"),  "application/octet-stream"),
        }

        # If only one file exists, return it directly
        existing_files = {name: info for name, info in output_files.items() if os.path.exists(info[0])}

        if len(existing_files) == 1:
            name, (path, media_type) = next(iter(existing_files.items()))
            return FileResponse(path, media_type=media_type, filename=name)

        # Otherwise, zip all existing outputs and return the zip
        zip_filename = f"{base_name}_Aligned.zip"
        zip_path = os.path.join(Config.OUTPUT_DIR, zip_filename)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, (path, _) in existing_files.items():
                zf.write(path, arcname=name)

        if not existing_files:
            raise HTTPException(status_code=500, detail="No output files were generated.")

        return FileResponse(zip_path, media_type="application/zip", filename=zip_filename)

@app.post("/batch/mfa")
async def batch_montreal_forced_aligner(
    file: UploadFile = File(..., description="ZIP file containing audio and JSON transcript pairs"),
    keyword: str = Form("INTEL", description="The keyword to filter out the files (e.g., study name)"),
    languages: Optional[str] = Form("english", description="Comma-separated languages. For example: english,spanish")
):
    """Batch process files through the Montreal Forced Aligner."""
    
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Please upload a .zip file containing batch.")

    # 1. Parse languages
    if languages:
        lang_map = {
            "english": Language.ENGLISH,
            "spanish": Language.SPANISH,
            "mandarin": Language.CHINESE,
        }
        selected_languages = [
            lang_map[lang.strip().lower()] 
            for lang in languages.split(",") 
            if lang.strip().lower() in lang_map
        ]
    else:
        selected_languages = Config.LANGUAGES

    with tempfile.TemporaryDirectory() as temp_dir:
        # 2. Setup internal input/output directories
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        # 3. Save and extract the uploaded ZIP file
        zip_input_path = os.path.join(temp_dir, file.filename)
        with open(zip_input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        with zipfile.ZipFile(zip_input_path, 'r') as zip_ref:
            zip_ref.extractall(input_dir)

        processed_count = 0

        # 4. Iterate through the extracted files using logic
        output_files = {}
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if keyword in filename and "parent_transcript_edit.json" in filename:
                    json_path = os.path.join(root, filename)
                    
                    # Split on '_' and append 'S.m4a' as defined in script
                    base_name = filename.split('_')[0]
                    audio_filename = f"{base_name}S.m4a" 
                    audio_path = os.path.join(root, audio_filename)

                    if not os.path.isfile(audio_path):
                        print(f"Skipped: Audio path {audio_path} doesn't exist for {json_path}")
                        continue

                    try:
                        # Convert M4A to WAV
                        wav_path = EP.m4a_to_wav(audio_path)
                        
                        # Load transcript
                        with open(json_path, 'r', encoding='utf-8') as f:
                            transcript = json.load(f)

                        # Call AP Script 
                        # Note: We pass `output_dir` so AP.script writes the results here
                        AP.script(
                            audio_path=wav_path, 
                            transcript=transcript, 
                            temp_dir=output_dir, 
                            languages=selected_languages, 
                            download_models=False
                        )
                        processed_count += 1
                        print(f"Successfully processed pair for: {base_name}")
                        
                    except Exception as e:
                        print(f"Failed to process {base_name}: {str(e)}")
                        # In batch processing, it's usually better to continue rather than crash the whole batch
                        continue
                    
                    base_name = os.path.basename(audio_path).split(".")[0]
                    output_files[f"{base_name}_Aligned.json"] = (os.path.join(Config.OUTPUT_DIR, f"{base_name}_Aligned.json"), "application/json")
                    output_files[f"{base_name}_Aligned.csv"] = (os.path.join(Config.OUTPUT_DIR, f"{base_name}_Aligned.csv"), "text/csv")
                    output_files[f"{base_name}_Aligned.TextGrid"] = (os.path.join(Config.OUTPUT_DIR, f"{base_name}_Aligned.TextGrid"),  "application/octet-stream")

        # 5. Check if anything actually processed
        if processed_count == 0:
            raise HTTPException(
                status_code=400, 
                detail="No matching file pairs were found or successfully processed in the ZIP."
            )

        # 6. Zip the results and save permanently to Config.OUTPUT_DIR
        batch_zip_filename = f"batch_mfa_results.zip"
        permanent_zip_path = os.path.join(Config.OUTPUT_DIR, batch_zip_filename)
        existing_files = {name: info for name, info in output_files.items() if os.path.exists(info[0])}

        with zipfile.ZipFile(permanent_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, (path, _) in existing_files.items():
                zf.write(path, arcname=name)
        if not existing_files:
            raise HTTPException(status_code=500, detail="No output files were generated.")

    # 7. Return the final ZIP
    return FileResponse(
        permanent_zip_path, 
        media_type="application/zip", 
        filename=batch_zip_filename
    )

# @app.post("/batch/m4a-to-wav")
# async def batch_convert_m4a_to_wav(
#     file: UploadFile = File(..., description="ZIP file containing M4A audio files to convert")
# ):
#     """Batch convert M4A files to WAV format from an uploaded ZIP archive."""
    
#     if not file.filename.lower().endswith('.zip'):
#         raise HTTPException(status_code=400, detail="Please upload a .zip file containing your batch.")

#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Setup internal input/output directories
#         input_dir = os.path.join(temp_dir, "input")
#         output_dir = os.path.join(temp_dir, "output")
#         os.makedirs(input_dir)
#         os.makedirs(output_dir)

#         # Save and extract the uploaded ZIP file
#         zip_input_path = os.path.join(temp_dir, file.filename)
#         with open(zip_input_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)

#         with zipfile.ZipFile(zip_input_path, 'r') as zip_ref:
#             zip_ref.extractall(input_dir)

#         processed_count = 0

#         # Iterate through the extracted files looking for .m4a
#         for root, _, files in os.walk(input_dir):
#             for filename in files:
#                 if filename.lower().endswith('.m4a'):
#                     m4a_path = os.path.join(root, filename)
                    
#                     try:
#                         # Convert M4A to WAV
#                         temp_wav_path = EP.m4a_to_wav(m4a_path)
                        
#                         # Maintain relative folder structure from the original zip
#                         rel_path = os.path.relpath(root, input_dir)
#                         out_subdir = os.path.join(output_dir, rel_path)
#                         os.makedirs(out_subdir, exist_ok=True)
                        
#                         # Move the generated wav to our structured output dir
#                         wav_filename = os.path.basename(temp_wav_path)
#                         wav_path = os.path.join(out_subdir, wav_filename)
#                         shutil.move(temp_wav_path, wav_path)
                        
#                         processed_count += 1
#                         print(f"Successfully converted: {filename}")
                        
#                     except Exception as e:
#                         print(f"Failed to process {filename}: {str(e)}")
#                         # Continue processing the rest of the batch even if one fails
#                         continue

#         # Check if anything actually processed
#         if processed_count == 0:
#             raise HTTPException(
#                 status_code=400, 
#                 detail="No valid M4A files were found or successfully converted in the ZIP."
#             )

#         # Zip the results and save permanently to Config.OUTPUT_DIR
#         batch_zip_filename = f"batch_m4a_to_wav_results.zip"
#         permanent_zip_path = os.path.join(Config.OUTPUT_DIR, batch_zip_filename)

#         with zipfile.ZipFile(permanent_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
#             for root, _, files in os.walk(output_dir):
#                 for out_file in files:
#                     file_path = os.path.join(root, out_file)
#                     # arcname keeps the zip clean by removing the absolute temp path
#                     arcname = os.path.relpath(file_path, output_dir)
#                     zf.write(file_path, arcname=arcname)

#     # Return the final ZIP
#     return FileResponse(
#         permanent_zip_path, 
#         media_type="application/zip", 
#         filename=batch_zip_filename
#     )

@app.get("/health")
async def health():
    return {"status": "healthy"}