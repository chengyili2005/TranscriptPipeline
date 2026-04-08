# Description

- - -

`src.py`: a script + a collection of helper functions to help with transcription process
- (TODO: list functions)

# Setup

0. Make a conda environment
```
conda create -n TranscriptPipeline python=3.11
conda activate TranscriptPipeline
```

1. Download requirements
```
pip install -r requirements.txt
```

# Assumptions

- Transcript json must be [{'start':start, 'end':end, 'text':text}, ...]
- Data is only English, Spanish, and/or Chinese (Mandarin).
- Scripts are being ran in the same directory as `TranscriptPipeline.py`

# Examples

Example: MFA alignment on TextGrid file

```python
import TranscriptPipeline as TP
from lingua import Language
from textgrid import TextGrid # For opening TextGrids

# Set variables & specify tier with utterances
audio_path = 'path/to/audio.wav'
textgrid_path = 'path/to/transcript.TextGrid'
utterance_tier = 0 # NOTE: First tier is utterance
languages = [Language.ENGLISH, Language.SPANISH, Language.CHINESE]
temp_dir = 'output/'

# Extract json from textgrid
tg = TextGrid()
tg.read(textgrid_path)
transcript = []
for interval in tg[utterance_tier]:
  transcript.append({
    "start": interval.minTime,
    "end": interval.maxTime,
    "text": interval.mark
  })

# Call script
segments = TP.script(audio_path=audio_path, transcript=transcript, temp_dir=temp_dir, languages=languages, download_models=False)
print(segments)
```

Example: Using on a single file

```python
import TranscriptPipeline as TP
from lingua import Language

# Set variables here
audio_path = 'path/to/audio.wav'
json_path = 'path/to/transcript.json'
languages = [Language.ENGLISH, Language.SPANISH, Language.CHINESE] # NOTE: Remove some languages if you know it's not in your data
temp_dir = 'output/'

# Call script
segments = TP.script(audio_path=audio_path, transcript=json_path, temp_dir=temp_dir, languages=languages, download_models=False)
print(segments)
```

Example: Batch align for multiple transcript jsons

```python
import TranscriptPipeline as TP
from lingua import Language

# Set variables as a list
audio_paths = ['path/to/audio1.wav', 'path/to/audio2.wav', 'path/to/audio3.wav']
json_paths = ['path/to/transcript1.json', 'path/to/transcript2.json', 'path/to/transcript3.json']
languages = [Language.ENGLISH, Language.SPANISH, Language.CHINESE]
temp_dir = 'output/'

# Call script iteratively
for audio_path, json_path in zip(audio_paths, json_paths): # NOTE: File paths must correspond to each other
  segments = TP.script(audio_path=audio_path, transcript=json_path, temp_dir=temp_dir, languages=languages, download_models=False)
  print(segments)
```