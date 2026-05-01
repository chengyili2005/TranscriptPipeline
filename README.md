# Setup

### Download dependencies
- [Git](https://git-scm.com/install/)
- [Conda](https://www.anaconda.com/download) (+ Python)

### Clone
```bash
# Open up a directory where you want to clone this folder. (Can be anywhere, just remember where it is for later.)
cd ~/Desktop/

# Pull my code from the cloud
git clone https://github.com/chengyili2005/TranscriptPipeline.git
```

### Make a conda environment w/ requirements
```bash
# Go inside the cloned directory
cd TranscriptPipeline/

# Create & activate the environment
conda env create -f environment.yaml
conda activate TranscriptPipeline
```

# Use as FastAPI

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Then, visit http://localhost:8000/docs for a frontend interface.

# Endpoints
You do not have to go through the full pipeline.
### Transcribe
- Takes in a `.wav` file and transcribes it using Whisper
    - If you already have a transcript, make sure it's in the assumed format in the **Assumptions** section. TextGrid2Json might help with that.
### Edit
- Json2TextGrid: After transcription, take the `.json` and convert it into a `.TextGrid`. Edit **ONLY** the **TEXT CONTENT**, no need to edit the timings as they will be fixed later.
- TextGrid2Json: After editting, turn the `.TextGrid` back into a `.json` for the alignment to be done.
### Align
- Takes in a `.wav` file + `.json` transcript and outputs a .zip file of aligned text in `.csv`, `.json`, and `.TextGrid`.
# Assumptions

- Transcript must be formatted like this:
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
- Scripts are being ran inside the repo directory.

Example: MFA alignment on TextGrid file

```python
import AlignPipeline as AP
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
segments = AP.script(audio_path=audio_path, transcript=transcript, temp_dir=output_dir, languages=languages, download_models=False)
print(segments)
```