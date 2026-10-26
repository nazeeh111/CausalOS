"""Generate CAUSALOS's original, deterministic 20-second score. No sampled media.

Run with Python and NumPy. The authored score and this source are covered by the
project MIT license. This is deliberately quiet; playback gain is set in HTML.
"""
from pathlib import Path
import json
import wave
import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "composition" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
RATE, DURATION = 44100, 20
t = np.arange(RATE * DURATION) / RATE
signal = np.zeros_like(t)


def tone(start, duration, frequency, gain, decay=1.5):
    select = (t >= start) & (t < start + duration)
    local = t[select] - start
    envelope = (1 - np.exp(-local * 24)) * np.exp(-local * decay)
    envelope *= np.minimum(1, (duration - local) * 12)
    sound = np.sin(2 * np.pi * frequency * local)
    sound += .22 * np.sin(2 * np.pi * 2 * frequency * local)
    sound += .06 * np.sin(2 * np.pi * 3 * frequency * local)
    signal[select] += gain * envelope * sound


# Original sparse harmonic progression, no borrowed melody or recordings.
chords = [(0, (146.832, 220, 293.665)), (4.5, (130.813, 195.998, 261.626)),
          (9, (164.814, 220, 329.628)), (15, (146.832, 220, 293.665, 440))]
for start, frequencies in chords:
    for i, frequency in enumerate(frequencies):
        tone(start + .06 * i, 5.0 if start < 15 else 4.9, frequency, .075, .55)
for start, frequency in ((1.1, 587.33), (3.0, 440), (5.7, 523.25), (7.6, 392),
                         (10.2, 659.25), (12.4, 440), (15.7, 587.33), (17.3, 880)):
    tone(start, 2.2, frequency, .047, 2.2)
for start in (8.8, 9.0):
    tone(start, .085, 1160 if start == 8.8 else 920, .09, 45)
signal *= np.minimum(1, t / .25) * np.minimum(1, (DURATION - t) / 1.4)
signal = np.clip(signal, -.95, .95)
with wave.open(str(ASSETS / "causalos-score.wav"), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(RATE)
    wav.writeframes((signal * 32767).astype("<i2").tobytes())

# Deterministic per-frame energy for a subtle identity-mark response.
frames = [float(np.sqrt(np.mean(signal[i*1470:(i+1)*1470] ** 2))) for i in range(600)]
peak = max(frames)
energy = [round(x / peak, 5) for x in frames]
(ASSETS / "score-energy.js").write_text("window.SCORE_ENERGY=" + json.dumps(energy) + ";\n")
(ASSETS / "score-cues.json").write_text(json.dumps({"duration": 20, "source": "original deterministic score",
    "strongCues": [{"time": x, "kind": "authored_harmonic_change"} for x in (4.5, 9, 15)],
    "interactionAccents": [8.8, 9.0]}, indent=2))
print(ASSETS / "causalos-score.wav")
