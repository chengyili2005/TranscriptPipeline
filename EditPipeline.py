import os
import json
from textgrid import TextGrid, IntervalTier
from pydub import AudioSegment

def m4a_to_wav(m4a_path):
    """Converts M4A file to WAV format, keeping the same basename."""
    wav_filename = m4a_path.replace('.m4a', '.wav')
    sound = AudioSegment.from_file(m4a_path, format='m4a')
    sound.export(wav_filename, format='wav')
    return wav_filename

def mp3_to_wav(mp3_path):
    """Converts MP3 file to WAV format, keeping the same basename."""
    wav_filename = mp3_path.replace('.mp3', '.wav')
    sound = AudioSegment.from_file(mp3_path, format='mp3')
    sound.export(wav_filename, format='wav')
    return wav_filename


def json_to_textgrid(json_data, output_path):
    """
    Converts a simple list of dicts: 
    [{'start': 0.0, 'end': 1.0, 'text': 'hello'}, ...] 
    into a TextGrid for Praat.
    """
    tg = TextGrid()
    # We create a single tier named 'edit_tier' to hold the text
    tier = IntervalTier(name='Edit This Tier!')
    
    for entry in json_data:
        # Basic validation to ensure the interval is valid for Praat
        if entry['end'] > entry['start']:
            tier.add(entry['start'], entry['end'], entry['text'])
    
    tg.append(tier)
    tg.write(output_path)
    return output_path

def textgrid_to_json(tg_path, tier_index=0):
    """
    Parses a specific tier from a TextGrid back into the simple JSON format.
    tier_index: 0-indexed integer specifying which tier to read.
    """
    tg = TextGrid()
    tg.read(tg_path)
    
    if tier_index >= len(tg):
        raise IndexError(f"Tier index {tier_index} out of range. File has {len(tg)} tiers.")

    target_tier = tg[tier_index]
    segments = []
    
    for interval in target_tier:
        # Praat often has empty intervals; we filter them out to keep the JSON clean
        if interval.mark.strip(): 
            segments.append({
                "start": round(interval.minTime, 4),
                "end": round(interval.maxTime, 4),
                "text": interval.mark.strip()
            })
            
    return segments