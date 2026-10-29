"""Original deterministic CAUSALOS score, adapted to the 24-second v0.1.1 edit.
No samples, borrowed recordings, recognizable melody or narration. MIT license.
"""
from pathlib import Path
import json
import wave
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'composition' / 'assets'
RATE, DURATION = 48000, 24

t = np.arange(RATE * DURATION) / RATE
signal = np.zeros_like(t)

def tone(start, duration, frequency, gain, decay=1.2):
    select = (t >= start) & (t < start + duration)
    local = t[select] - start
    envelope = (1 - np.exp(-local * 24)) * np.exp(-local * decay)
    envelope *= np.minimum(1, (duration - local) * 12)
    sound = np.sin(2 * np.pi * frequency * local)
    sound += .18 * np.sin(2 * np.pi * 2 * frequency * local)
    sound += .04 * np.sin(2 * np.pi * 3 * frequency * local)
    signal[select] += gain * envelope * sound

chords = [(0, (146.832, 220, 293.665), 4.0),
          (3.6, (130.813, 195.998, 261.626), 5.3),
          (8.5, (164.814, 220, 329.628), 6.5),
          (15, (110, 164.814, 246.942), 4.8),
          (19.5, (146.832, 220, 293.665, 440), 4.5)]
for start, frequencies, duration in chords:
    for i, frequency in enumerate(frequencies):
        tone(start + .05 * i, duration - .05 * i, frequency, .075, .50)
for start, frequency in ((1.0, 587.33), (2.7, 440), (5.0, 523.25), (7.2, 392),
                         (9.7, 659.25), (12.5, 440), (16.1, 493.883),
                         (20.2, 587.33), (22.0, 880)):
    tone(start, 1.8, frequency, .038, 2.1)
# Quiet accents mark edited action/result handoffs, not real-time experiment execution.
for start, frequency in ((3.85, 1040), (8.5, 880)):
    tone(start, .075, frequency, .045, 48)
signal *= np.minimum(1, t / .25) * np.minimum(1, (DURATION - t) / 1.6)
assert np.max(np.abs(signal)) < .95
ASSETS.mkdir(parents=True, exist_ok=True)
with wave.open(str(ASSETS / 'score-original.wav'), 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(RATE)
    wav.writeframes((signal * 32767).astype('<i2').tobytes())
(ASSETS / 'score-cues.json').write_text(json.dumps({'duration':DURATION,
    'source':'original deterministic synthesis','scene_changes':[3.6,8.5,15,19.5],
    'interaction_accents':[3.85,8.5],'voice':False},indent=2)+'\n')
print(ASSETS / 'score-original.wav')
