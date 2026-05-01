import torch
import json
import subprocess
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from faster_whisper import WhisperModel
from vosk import Model, KaldiRecognizer

HUB_REPO = "chengyili2005/whisper-medium-DINA"
BASE_MODEL = "openai/whisper-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Specify the language for auto-download, or use an exact model name
# via model_name="vosk-model-en-us-0.22" inside the Model() call.
VOSK_MODEL_LANG = "en-us"

# ── Global Cache for Models ──────────────────────────────────────────────────
_asr_pipeline = None  # For PEFT / Transformers
_faster_model = None  # For Faster-Whisper
_vosk_model = None  # For Vosk

# ── Loaders ──────────────────────────────────────────────────────────────────
def _load_whisper_peft():
    global _asr_pipeline
    if _asr_pipeline is not None:
        return _asr_pipeline

    processor = WhisperProcessor.from_pretrained(
        HUB_REPO, language="english", task="transcribe"
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, HUB_REPO)
    model = model.merge_and_unload()
    model.to(DEVICE)
    model.eval()

    _asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch.float16,
        device=DEVICE,
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=1,
    )
    return _asr_pipeline


def _load_faster_whisper():
    global _faster_model
    if _faster_model is not None:
        return _faster_model

    compute_type = "float16" if DEVICE == "cuda" else "int8"
    _faster_model = WhisperModel(
        "small", device=DEVICE, compute_type=compute_type
    )
    return _faster_model


def _load_vosk():
    global _vosk_model
    if _vosk_model is not None:
        return _vosk_model

    # Passing 'lang' or 'model_name' forces Vosk to check its local cache.
    # If the model isn't found, it downloads it on the fly.
    print(
        f"Loading Vosk model (language: {VOSK_MODEL_LANG}). "
        "It will download automatically if not cached..."
    )
    _vosk_model = Model(lang=VOSK_MODEL_LANG)

    return _vosk_model


# ── Utterance grouping ───────────────────────────────────────────────────────
def _group_into_utterances(
    word_chunks: list,
    max_pause_s: float = 1.0,
    max_duration_s: float = 8.0,
) -> list[dict]:
    utterances = []
    current_words = []
    current_start = None
    current_end = None

    for chunk in word_chunks:
        word = chunk["text"]
        start, end = chunk["timestamp"]

        if start is None or end is None:
            continue

        if current_start is None:
            current_start = start
            current_end = end
            current_words.append(word)
        else:
            pause = start - current_end
            duration = end - current_start

            if pause > max_pause_s or duration > max_duration_s:
                utterances.append(
                    {
                        "start": round(current_start, 3),
                        "end": round(current_end, 3),
                        "text": "".join(current_words).strip(),
                    }
                )
                current_start = start
                current_words = [word]
            else:
                current_words.append(word)

            current_end = end

    if current_words:
        utterances.append(
            {
                "start": round(current_start, 3),
                "end": round(current_end, 3),
                "text": "".join(current_words).strip(),
            }
        )

    return utterances


# ── Transcription Logic ──────────────────────────────────────────────────────
def _transcribe_peft(audio_path: str) -> list[dict]:
    asr = _load_whisper_peft()

    try:
        with torch.inference_mode():
            result = asr(
                audio_path,
                generate_kwargs={
                    "language": "english",
                    "task": "transcribe",
                    "condition_on_prev_tokens": False,
                    "no_speech_threshold": 0.6,
                    "logprob_threshold": -1.0,
                    "compression_ratio_threshold": 1.35,
                    "temperature": 0.0,
                },
                return_timestamps="word",
            )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _group_into_utterances(result["chunks"])


def _transcribe_faster(audio_path: str) -> list[dict]:
    model = _load_faster_whisper()

    segments, _ = model.transcribe(
        audio_path, beam_size=5, word_timestamps=True, language="en"
    )

    # Format faster-whisper word objects to match the _group_into_utterances schema
    word_chunks = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                word_chunks.append(
                    {"text": w.word, "timestamp": (w.start, w.end)}
                )

    return _group_into_utterances(word_chunks)


def _transcribe_vosk(audio_path: str) -> list[dict]:
    model = _load_vosk()

    # Vosk requires 16kHz mono audio. FFmpeg converts this on the fly so we
    # don't have to manually format the audio file beforehand.
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-i",
            audio_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    word_chunks = []

    # Read audio chunks and feed them to Vosk
    while True:
        data = process.stdout.read(4000)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if "result" in res:
                for w in res["result"]:
                    # We add a leading space to match Whisper's spacing behavior
                    # so the grouping function handles concatenation perfectly.
                    word_chunks.append(
                        {"text": " " + w["word"], "timestamp": (w["start"], w["end"])}
                    )

    # Process the final remaining chunk
    final_res = json.loads(rec.FinalResult())
    if "result" in final_res:
        for w in final_res["result"]:
            word_chunks.append(
                {"text": " " + w["word"], "timestamp": (w["start"], w["end"])}
            )

    return _group_into_utterances(word_chunks)


# ── Public entry point ───────────────────────────────────────────────────────
def script(
    audio_path: str, temp_dir: str, method: str, **kwargs
) -> list[dict]:
    """
    Transcribe an audio file and return a list of utterance dicts.

    Each dict has the shape:
        { "start": float, "end": float, "text": str }

    Args:
        audio_path: Path to the input audio file.
        temp_dir:   Temporary working directory (reserved for future use).
        method:     Transcription backend ("whisper", "faster-whisper", or "vosk").

    Returns:
        List of utterance dicts sorted by start time.
    """
    method = method.lower()

    if method == "whisper":
        return _transcribe_peft(audio_path)
    elif method == "faster-whisper":
        return _transcribe_faster(audio_path)
    elif method == "vosk":
        return _transcribe_vosk(audio_path)
    elif method == "aws":
        raise NotImplementedError("AWS transcription is not yet implemented")
    elif method == "sonix":
        raise NotImplementedError("Sonix transcription is not yet implemented")
    else:
        raise ValueError(
            f"Unknown transcription method: '{method}'. "
            "Currently only 'whisper', 'faster-whisper', and 'vosk' are supported."
        )