from lingua import Language

# TODO

# result = TP.script(
#     audio_path=audio_path,
#     temp_dir=temp_dir,
#     languages=Config.LANGUAGES,
#     method=method,
# )

# Meta info:
#   speaker diarization -> json format
 
# Transcription choice:
#   transcribe locally -> json format
#   transcribe through AWS API -> json format
#   transcribe through SONIX API -> json format

def script(audio_path: str, temp_dir: str, languages: list[Language], method: str):
    return {} 