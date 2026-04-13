# Description

- - -

`TranscriptPipeline.py`: a script + a collection of helper functions to help with transcription process
- (TODO: list functions)

# Dependencies
- [Git](https://git-scm.com/install/)
- [Conda](https://www.anaconda.com/download) (+ Python)

# Setup

### Clone
```bash
git clone https://github.com/chengyili2005/TranscriptPipeline.git
```

### Make a conda environment w/ requirements
```bash
# Create & activate the environment
conda env create -f environment.yml
conda activate TranscriptPipeline
```

# Assumptions

- Transcript json must be formatted like this:
```python
[
  {'start':start, 'end':end, 'text':text},
  {'start':start, 'end':end, 'text':text}, 
  ...
  {'start':start, 'end':end, 'text':text}
] 
# See examples for how to get this from a TextGrid
```
- Data is only English, Spanish, and/or Chinese (Mandarin).
- Scripts are being ran in the same directory as `TranscriptPipeline.py`

# Examples

Example: MFA alignment on TextGrid file

```python
import TranscriptPipeline as TP
from lingua import Language
from textgrid import TextGrid # For opening TextGrids

# Set variables & specify tier with utterances
audio_path = 'input/example1.wav'
textgrid_path = 'input/example1.TextGrid'
utterance_tier = 0 # NOTE: First tier is utterance
languages = [Language.ENGLISH, Language.SPANISH, Language.CHINESE]
output_dir = 'output/'

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
segments = TP.script(audio_path=audio_path, transcript=transcript, temp_dir=output_dir, languages=languages, download_models=False)
print(segments)
```

Example: Using on a single file

```python
import TranscriptPipeline as TP
from lingua import Language

# Set variables here
audio_path = 'input/example1.wav'
json_path = 'input/example1.json'
languages = [Language.ENGLISH, Language.SPANISH, Language.CHINESE] # NOTE: Remove some languages if you know it's not in your data
output_dir = 'output/'

# Call script
segments = TP.script(audio_path=audio_path, transcript=json_path, temp_dir=output_dir, languages=languages, download_models=False)
print(segments)
```

Example: Batch align for multiple transcript jsons

```python
import TranscriptPipeline as TP
from lingua import Language

# Set variables as a list
audio_paths = ['input/example1.wav', 'input/example2.wav', 'input/example3.wav']
json_paths = ['input/transcript1.json', 'input/transcript2.json', 'input/transcript3.json']
languages = [Language.ENGLISH, Language.SPANISH, Language.CHINESE]
output_dir = 'output/'

# Call script iteratively
for audio_path, json_path in zip(audio_paths, json_paths): # NOTE: File paths must correspond to each other
  segments = TP.script(audio_path=audio_path, transcript=json_path, temp_dir=output_dir, languages=languages, download_models=False)
  print(segments)
```