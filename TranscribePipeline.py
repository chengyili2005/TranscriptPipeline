import torch
import json
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

HUB_REPO   = "chengyili2005/whisper-medium-DINA"
BASE_MODEL = "openai/whisper-medium"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model loader (cached so repeated calls don't reload) ───────────────────────
_asr_pipeline = None

def _load_whisper():
    global _asr_pipeline
    if _asr_pipeline is not None:
        return _asr_pipeline

    processor  = WhisperProcessor.from_pretrained(HUB_REPO, language="english", task="transcribe")
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


# ── Utterance grouping ─────────────────────────────────────────────────────────
def _group_into_utterances(
    word_chunks: list,
    max_pause_s: float = 1.0,
    max_duration_s: float = 8.0,
) -> list[dict]:
    utterances    = []
    current_words = []
    current_start = None
    current_end   = None

    for chunk in word_chunks:
        word       = chunk["text"]
        start, end = chunk["timestamp"]

        if start is None or end is None:
            continue

        if current_start is None:
            current_start = start
            current_end   = end
            current_words.append(word)
        else:
            pause    = start - current_end
            duration = end - current_start

            if pause > max_pause_s or duration > max_duration_s:
                utterances.append({
                    "start": round(current_start, 3),
                    "end":   round(current_end,   3),
                    "text":  "".join(current_words).strip(),
                })
                current_start = start
                current_words = [word]
            else:
                current_words.append(word)

            current_end = end

    if current_words:
        utterances.append({
            "start": round(current_start, 3),
            "end":   round(current_end,   3),
            "text":  "".join(current_words).strip(),
        })

    return utterances


# ── Whisper transcription ──────────────────────────────────────────────────────
def _transcribe_whisper(audio_path: str) -> list[dict]:
    asr = _load_whisper()

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


# ── Public entry point ─────────────────────────────────────────────────────────
def script(audio_path: str, temp_dir: str, method: str, **kwargs) -> list[dict]:
    """
    Transcribe an audio file and return a list of utterance dicts.

    Each dict has the shape:
        { "start": float, "end": float, "text": str }

    Args:
        audio_path: Path to the input audio file.
        temp_dir:   Temporary working directory (reserved for future use).
        method:     Transcription backend — only "whisper" is supported right now.

    Returns:
        List of utterance dicts sorted by start time.
    """
    if method == "whisper":
        return _transcribe_whisper(audio_path)
    elif method == "aws":
        raise NotImplementedError("AWS transcription is not yet implemented")
    elif method == "sonix":
        raise NotImplementedError("Sonix transcription is not yet implemented")
    else:
        raise ValueError(f"Unknown transcription method: '{method}'. Currently only 'whisper' is supported.")