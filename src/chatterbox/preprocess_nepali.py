"""
Preprocess Nepali audio data for training:
1. Read transcriptions.csv and map to actual WAV files
2. Resample to 24000 Hz
3. Filter by duration (2-10 seconds)
4. Generate manifest.jsonl + organize into data/wavs/
"""
import argparse
import csv
import json
import os
from pathlib import Path
import librosa
import soundfile as sf

TARGET_SR = 24000
MIN_DUR = 1.0  # seconds
MAX_DUR = 15.0  # seconds

def preprocess(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"

    transcriptions = []
    with open(data_dir / "transcriptions.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcriptions.append(row)

    print(f"📖 Loaded {len(transcriptions)} transcriptions")

    skipped_no_file = 0
    skipped_duration = 0
    saved = 0

    manifest_entries = []

    for i, row in enumerate(transcriptions):
        name = row["name"]
        text = row["transcription"]

        # Resolve audio path
        # Examples:
        #   "1 /Company/20260220 160847.wav" -> data/1 /Company/20260220 160847.wav
        #   "2/20260220 165622.wav"         -> data/2/20260220 165622.wav
        #   "Recording (100)_chunk_0000.wav" -> data/Recording (100)_chunk_0000.wav
        audio_path = data_dir / name
        if not audio_path.exists():
            skipped_no_file += 1
            if skipped_no_file <= 5:
                print(f"⚠️  File not found: {audio_path}")
            continue

        # Load and resample
        try:
            wav, sr = librosa.load(str(audio_path), sr=None)
        except Exception as e:
            print(f"⚠️  Failed to load {audio_path}: {e}")
            continue

        # Resample if needed
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

        duration = len(wav) / TARGET_SR

        # Duration filter
        if duration < MIN_DUR or duration > MAX_DUR:
            skipped_duration += 1
            if skipped_duration <= 5:
                print(f"⏭️  Skipped (dur={duration:.2f}s): {name}")
            continue

        # Save as 24kHz WAV
        # Create a safe filename
        safe_name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_wav = wav_dir / f"{safe_name}"
        sf.write(str(out_wav), wav, TARGET_SR, subtype="PCM_16")

        manifest_entries.append({
            "audio_path": str(out_wav),
            "text": text
        })
        saved += 1

    # Write manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Done!")
    print(f"   Saved:     {saved}")
    print(f"   No file:   {skipped_no_file}")
    print(f"   Bad dur:   {skipped_duration}")
    print(f"   Manifest:  {manifest_path}")

    # Duration stats
    if manifest_entries:
        durations = []
        for entry in manifest_entries:
            wav, _ = librosa.load(entry["audio_path"], sr=TARGET_SR)
            durations.append(len(wav) / TARGET_SR)
        import numpy as np
        print(f"\n   📊 Duration stats:")
        print(f"      Min: {np.min(durations):.2f}s")
        print(f"      Max: {np.max(durations):.2f}s")
        print(f"      Mean: {np.mean(durations):.2f}s")
        print(f"      Total: {sum(durations)/3600:.2f} hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="data/nepali")
    args = parser.parse_args()
    preprocess(args)
