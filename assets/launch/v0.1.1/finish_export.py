"""Losslessly remove the export capture margin and make a streamable MP4.

The authored picture is 1920x1080. index-overscan.html adds 200px beneath it
for the installed renderer's clipped viewport. H.264 picture cropping changes
only display metadata; the remaining decoded pixels and AAC stream are retained.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('render',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists() or args.output.is_symlink():
    raise SystemExit('Choose a fresh output; existing masters are preserved.')
ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
if not ffmpeg or not ffprobe:
    raise SystemExit('Installed FFmpeg and FFprobe are required.')
probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams','-of','json',str(args.render)]))
v=next(s for s in probe['streams'] if s['codec_type']=='video')
if (v['codec_name'],v['width'],v['height'])!=('h264',1920,1280):
    raise SystemExit('Expected the 1920x1280 H.264 capture-margin render.')
subprocess.run([ffmpeg,'-hide_banner','-nostdin','-v','error','-i',str(args.render),'-map','0','-c','copy',
    '-bsf:v','h264_metadata=crop_bottom=200','-movflags','+faststart',
    '-metadata','title=CAUSALOS v0.1.1 · Evidence before conclusions',
    '-metadata','comment=Genuine v0.1.1 application captures. Simulated finite-model interventions; no real-world causal verification.',
    str(args.output)],check=True)
print(args.output)
