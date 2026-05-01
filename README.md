# Setup

### Download dependencies
- [Git](https://git-scm.com/install/)
- [Conda](https://www.anaconda.com/download) (+ Python)

### Clone
```bash
git clone https://github.com/chengyili2005/TranscriptPipeline.git
```

### Make a conda environment w/ requirements
```bash
# Create & activate the environment
conda env create -f environment.yaml
conda activate TranscriptPipeline
```

# Use as FastAPI

```bash
# cd TranscriptPipeline/ directory
uvicorn api:app --host 0.0.0.0 --port 8000
```
Then, visit http://localhost:8000/docs for a frontend interface

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
