"""Reproduce the restrained two-pass soundtrack mix with an installed FFmpeg.
Run after generate_audio.py. Refuses to replace an existing output.
"""
from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=ROOT/'composition/assets/score-mix-new.wav')
args = parser.parse_args()
if args.output.exists() or args.output.is_symlink():
    raise SystemExit('Choose a fresh output path; existing audio is preserved.')
ffmpeg = shutil.which('ffmpeg')
if not ffmpeg:
    raise SystemExit('An installed FFmpeg is required.')
source = ROOT/'composition/assets/score-original.wav'
first = subprocess.run([ffmpeg,'-hide_banner','-nostdin','-i',str(source),'-af',
    'loudnorm=I=-20:TP=-2:LRA=10:print_format=json','-f','null','-'],check=True,capture_output=True,text=True)
measure = json.loads(re.search(r'\{\s*"input_i".*?\}',first.stderr,re.S).group())
filters = f"loudnorm=I=-20:TP=-2:LRA=10:measured_I={measure['input_i']}:measured_TP={measure['input_tp']}:measured_LRA={measure['input_lra']}:measured_thresh={measure['input_thresh']}:offset={measure['target_offset']}:linear=true:print_format=json"
subprocess.run([ffmpeg,'-hide_banner','-nostdin','-i',str(source),'-af',filters,'-ar','48000','-ac','2',str(args.output)],check=True)
print(args.output)
