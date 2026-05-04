"""
NOTES:

Two main directories will be used:
- `~/Documents/MFA` is the MFA config directory. I can't change this no matter how much I try... This will be where downloaded models and cache will be stored
- `~/{working_dir}/./cache` is the cache directory. This is where temporary TextGrids will be read and written

"""

import os
import shutil
import subprocess as sp
from lingua import Language, LanguageDetectorBuilder
from textgrid import TextGrid, IntervalTier
import json
import pandas as pd
import ConfigPipeline as Config

# === GLOBALS === #

OUTPUT_DIR = Config.OUTPUT_DIR
LANGUAGES = Config.LANGUAGES
LANGUAGES_MAP = {
  Language.ENGLISH: {'dictionary': 'english_us_arpa', 'acoustic': 'english_us_arpa'},
  Language.SPANISH: {'dictionary': 'spanish_mfa', 'acoustic': 'spanish_mfa'},
  Language.CHINESE: {'dictionary': 'mandarin_mfa', 'acoustic': 'mandarin_mfa'},
}
DOWNLOADED_ACOUSTICS = []
DOWNLOADED_DICTIONARIES = []

# === CONFIGURE MFA === #

def configure_MFA_settings():
  out = sp.run(['mfa', 'configure', '--always_clean'], capture_output=True)
  if out.returncode != 0:
    print("Error Output:", out)
    raise Exception(f"Failed to configure")
  out = sp.run(['mfa', 'configure', '--always_final_clean'], capture_output=True)
  if out.returncode != 0:
    print("Error Output:", out)
    raise Exception(f"Failed to configure")

# === DOWNLOAD MODELS === # (Only need to do once!)

def download_MFA_models():
  for language in LANGUAGES_MAP:

    acoustic_name, dictionary_name = LANGUAGES_MAP[language]['acoustic'], LANGUAGES_MAP[language]['dictionary']

    if acoustic_name not in DOWNLOADED_ACOUSTICS:
      out = sp.run(['mfa', 'model', 'download', 'acoustic', acoustic_name], capture_output=True)
      if out.returncode != 0:
          print("Error:", out)
          raise Exception(f"Could not download acoustic model {acoustic_name}")
      DOWNLOADED_ACOUSTICS.append(acoustic_name)

    if dictionary_name not in DOWNLOADED_DICTIONARIES:
      out = sp.run(['mfa', 'model', 'download', 'dictionary', dictionary_name], capture_output=True)
      if out.returncode != 0:
          print("Error:", out)
          raise Exception(f"Could not download acoustic model {dictionary_name}")
      DOWNLOADED_DICTIONARIES.append(dictionary_name)

# === OTHER HELPER FUNCTIONS === #

def detect_language(text, detector=LanguageDetectorBuilder.from_languages(*LANGUAGES).build()):
  """Detects a language using lingua. text -> [lingua.Language, float]"""
  language, confidence = Language.ENGLISH, 0.0
  if text.isdigit(): # For numbers we really can't give a good guess
    return language, confidence
  elif any(char in text for char in "¡¿áéíóúñüÁÉÍÓÚÑÜ"): # Through special Spanish characters
      language, confidence = Language.SPANISH, 1.0
  else: # Through lingua
    language = detector.detect_language_of(text)
    confidence = detector.compute_language_confidence(text, language)
  return language, confidence

def is_empty_textgrid(tg):
  """Returns True if textgrid is empty"""
  is_empty = True
  if len(tg) >= 1:
    for index in range(len(tg)):
      if len(tg[index]) >= 1:
        is_empty = False
        return is_empty
  return is_empty

def contains_language(segments, language):
  """Returns True if you pass in a list of segments and the passed in language is in there"""
  for segment in segments:
    if segment['language'][0] == language:
      return True
  return False

def done2textgrid(segments, output_path=os.path.join(OUTPUT_DIR, 'base_name' + '_Aligned' + '.TextGrid')):
    """Returns the final transcript (AFTER RUNNING THE FULL SCRIPT) into a TextGrid"""
    tg = TextGrid()
    raw_utt_tier = IntervalTier(name="Raw Utterance", minTime=tg.minTime, maxTime=tg.maxTime)
    aligned_utt_tier = IntervalTier(name="Aligned Utterance", minTime=tg.minTime, maxTime=tg.maxTime)
    utt_lang_tier = IntervalTier(name="Utterance Language", minTime=tg.minTime, maxTime=tg.maxTime)
    word_tier = IntervalTier(name="MFA Word", minTime=tg.minTime, maxTime=tg.maxTime)
    word_lang_tier = IntervalTier(name="Word Languages", minTime=tg.minTime, maxTime=tg.maxTime)

    for segment in segments:
        try:
            raw_utt_tier.add(segment['start'], segment['end'], segment['text'])
        except Exception as e:
            print("Failed to add raw_utt_tier:", e, segment)

        words = segment.get('words', [])

        seen = set()
        unique_words = []
        for w in words:
            key = (w['start'], w['end'], w['text'])
            if key not in seen:
                seen.add(key)
                unique_words.append(w)

        if unique_words:
            try:
                aligned_utt_tier.add(
                    unique_words[0]['start'],
                    unique_words[-1]['end'],
                    segment['text']
                )
            except Exception as e:
                print("Failed to add aligned_utt_tier:", e, segment)
            try:
                utt_lang_tier.add(
                    unique_words[0]['start'],
                    unique_words[-1]['end'],
                    f"{segment['language'][0]}, {segment['language'][1]:.4f}"
                )
            except Exception as e:
                print("Failed to add utt_lang_tier:", e, segment)
        else:
            try:
                aligned_utt_tier.add(segment['start'], segment['end'], segment['text'])
            except Exception as e:
                print("Failed to add aligned_utt_tier (fallback):", e, segment)
            try:
                utt_lang_tier.add(
                    segment['start'],
                    segment['end'],
                    f"{segment['language'][0]}, {segment['language'][1]:.4f}"
                )
            except Exception as e:
                print("Failed to add utt_lang_tier (fallback):", e, segment)

        for word_segment in unique_words:
            try:
                word_tier.add(word_segment['start'], word_segment['end'], word_segment['text'])
            except Exception as e:
                print("Failed to add word_tier:", e, word_segment)
            try:
                word_lang_tier.add(
                    word_segment['start'],
                    word_segment['end'],
                    f"{word_segment['language'][0]}, {word_segment['language'][1]:.4f}"
                )
            except Exception as e:
                print("Failed to add word_lang_tier:", e, word_segment)

    tg.append(raw_utt_tier)
    tg.append(aligned_utt_tier)
    tg.append(utt_lang_tier)
    tg.append(word_tier)
    tg.append(word_lang_tier)

    if not is_empty_textgrid(tg):
        tg.write(output_path)
    else:
        raise Exception("Empty TextGrid")
    print("Exported to: ", output_path)

def done2json(segments, output_path=os.path.join(OUTPUT_DIR, 'base_name' + '_Aligned' + '.json')):
  """Exports to json"""
  with open(output_path, 'w') as f:
    json.dump(segments, f, indent=4)
  print("Exported to: ", output_path)

def done2csv(segments, output_path=os.path.join(OUTPUT_DIR, 'base_name' + '_Aligned' + '.csv')):
  """Exports to pandas dataframe csv"""
  df = []
  for segment in segments:
    if len(segment['words']) == 0:
       start = segment['start']
       end = segment['end']
    else:
       start = segment['words'][0]['start']
       end = segment['words'][-1]['end']
    df.append({
      'start': start,
      'end': end,
      'text': segment['text'],
      'language': segment['language'],
      'words': segment['words']
    })
  df = pd.DataFrame(df)
  df.to_csv(output_path, index=False)
  print("Exported to: ", output_path)

# === SCRIPT === #

def script(audio_path=str, transcript=str or list, temp_dir=OUTPUT_DIR, languages=list[Language], download_models=False):

  # Download
  if download_models:
    download_MFA_models()
    print("Downloaded MFA models. You can set download_models to False next time to skip this step!")

  # Configure MFA
  configure_MFA_settings()
  print("MFA configured with --always_clean and --always_final_clean. You can change this in the configure_MFA_settings function.")

  # Get variables
  base_name = os.path.basename(audio_path).split('.')[0]

  # Load in the transcript
  if '.json' in transcript:
    with open(transcript, 'r') as f:
      segments = json.load(f)
  else:
     segments = transcript

  # Detect a language for each utterance
  for segment in segments:
    segment['language'] = detect_language(segment['text'])

  # Preparing: Make separate temporary directories for each language
  os.makedirs(temp_dir, exist_ok=True)
  for language in languages:

    # Make directories
    language_dir = os.path.join(temp_dir, language.name)
    language_in_dir = os.path.join(language_dir, 'input')
    language_out_dir = os.path.join(language_dir, 'output')
    os.makedirs(language_dir, exist_ok=True)
    os.makedirs(language_in_dir, exist_ok=True)
    os.makedirs(language_out_dir, exist_ok=True)

    # Make a textgrid with only a specified language's utterances
    tg = TextGrid()
    text_tier = IntervalTier(name='text', minTime=tg.minTime, maxTime=tg.maxTime)
    for segment in segments:
      if segment['language'][0] == language:
        try:
          text_tier.add(segment['start'], segment['end'], segment['text'])
        except Exception as e:
          print("Failed to append", segment, e)
    tg.append(text_tier)

    # Populate the input directory with that language textgrid + a copy of the audio
    shutil.copy2(audio_path, language_in_dir)
    if not is_empty_textgrid(tg):
      tg.write(os.path.join(language_in_dir, base_name + '.TextGrid'))

    # Keep paths for later
    LANGUAGES_MAP[language]['language_dir'] = language_dir
    LANGUAGES_MAP[language]['language_in_dir'] = language_in_dir # MFA's corpus directory
    LANGUAGES_MAP[language]['language_out_dir'] = language_out_dir # MFA's output directory
    print(f"Prepared MFA input for {language}. You can check {language_in_dir} to see the TextGrid and audio that MFA will be aligning for this language.")

  """
  At this point: {TEMP_DIR}

  - LANGUAGE_1/
    - input/                          <- Contains LANGUAGE_1 utterances .TextGrid + copy of .wav file
      - file.wav
      - file.TextGrid
    - output/
  - LANGUAGE_2/
    - input/                          <- Contains LANGUAGE_2 utterances .TextGrid + copy of .wav file
      - file.wav
      - file.TextGrid
    - output/
  ...

  """

  # Alignment: Run MFA on these files
  word_intervals = []
  for language in LANGUAGES:
    if contains_language(segments, language):
      acoustic_name, dictionary_name = LANGUAGES_MAP[language]['acoustic'], LANGUAGES_MAP[language]['dictionary']
      language_dir, language_in_dir, language_out_dir = LANGUAGES_MAP[language]['language_dir'], LANGUAGES_MAP[language]['language_in_dir'], LANGUAGES_MAP[language]['language_out_dir']
      print(f"{language} alignment start.")
      command = ['mfa', 'align', language_in_dir, dictionary_name, acoustic_name, language_out_dir]
      print(f"Running: {' '.join(command)}")
      out = sp.run(command, capture_output=True)
      if out.returncode != 0:
        print("Error Output:", out)
        raise Exception(f"Failed to align for {language}")
      print(f"{language} alignment done.")
      tg = TextGrid()
      tg.read(os.path.join(language_out_dir, base_name + '.TextGrid'))
      word_intervals += [interval for interval in tg[0]]

  # Postprocessing: Sorting words by start time
  word_intervals = [{"start": interval.minTime, "end": interval.maxTime, "text": interval.mark} for interval in word_intervals if interval.mark.strip() != ""]
  word_intervals = sorted(word_intervals, key=lambda x: x['start'])

  # Postprocessing: Combining with utterances
  for segment in segments:
    segment['words'] = []
    for word_interval in word_intervals:
      if word_interval['start'] >= segment['start'] - 0.01 and word_interval['end'] <= segment['end'] + 0.01:
        segment['words'].append(word_interval)

  # Extra: Utterance languages are already appended onto each segment.

  # Extra: Add word languages
  for segment in segments:
    utt_lang, utt_conf = segment['language']
    for word in segment['words']:
      word_lang, word_conf = detect_language(word['text'])
      if utt_lang == word_lang:
        word['language'] = utt_lang, max(utt_conf, word_conf)
      else:
        if utt_conf > word_conf:
          word['language'] = utt_lang, word_conf
        else:
          word['language'] = word_lang, utt_conf

  # Postprocessing: Turn languages into strings
  for segment in segments:
    segment['language'] = segment['language'][0].name, segment['language'][1]
    for word_segment in segment['words']:
      word_segment['language'] = word_segment['language'][0].name, segment['language'][1]

  # Export: Praat
  print("Start exporting TextGrid...")
  done2textgrid(segments, output_path=os.path.join(OUTPUT_DIR, base_name + '_Aligned' + '.TextGrid'))
  print("Done exporting TextGrid.")

  # Export: Json
  print("Start exporting json...")
  done2json(segments, output_path=os.path.join(OUTPUT_DIR, base_name + '_Aligned' + '.json'))
  print("Done exporting json.")

  # Export: CSV
  print("Start exporting CSV...")
  done2csv(segments, output_path=os.path.join(OUTPUT_DIR, base_name + '_Aligned' + '.csv'))
  print("Done exporting CSV.")

  # Export: Datavyu
  # NOTE: Exporting to Datavyu is near-impossible without Datavyu's environment so I'm just going to leave it here for exports...

  return segments

# Example script usage
if __name__ == "__main__":
   
  # Set variables
  INPUT_DIR = 'input/'
  audio_path = os.path.join(INPUT_DIR, 'example.wav')
  transcript_path = os.path.join(INPUT_DIR, 'example.json') # NOTE: If it's not in a [{"start": float, "end": float, "text": str}, {"start": float, "end": float, "text": str}, ..., {"start": float, "end": float, "text": str}] format, you will need to do some extra processing to get it into this format
  
  # Run alignment
  segments = script(audio_path, transcript_path, temp_dir=OUTPUT_DIR, languages=LANGUAGES, download_models=False)
  print("Done! Segments:", segments)
  print("Also saved as TextGrid, CSV, and json in the output directory.") # More examples can be found in the README