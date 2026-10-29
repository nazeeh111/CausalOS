# v0.1.1 launch film verification

Selected master: `causalos-v0.1.1-launch.mp4` (SHA256 `cfc60d13db82445618464adada5948aaa9566fd4c82d46cfbe50a74ada149ffd`). Measured properties and exact source/capture hashes are in `media-verification.json` and `evidence.json`.

- 24 seconds, 720 frames, 1920×1080 display, 30 fps, H.264 yuv420p and 48 kHz stereo AAC. Full FFmpeg decode passes. Native browser playback reached the end at normal speed with controls and unmuted state.
- Actual encoded experiment, evidence, assumption and outro frames were inspected. Lower scientific captions and links are present. The installed renderer's clipped lower viewport was corrected with a 200-pixel capture margin and lossless H.264 display cropping. At 12 seconds, final pixels exactly match the top 1080 rows of the capture. A second stream-copy pass refreshes MP4 track/sample dimensions to 1920×1080 before the first decoded frame. Encoded video and audio hashes remain identical to the preceding corrected export. The finalizer reproduces the master byte-for-byte.
- The AAC stream is byte-identical to the user-selected creative cut. Integrated loudness is −20.01 LUFS, true peak −8.15 dBTP; no clipping. Measured correlation with the mix is 0.999973 with zero alignment lag. Original music and two-pass mix sources are retained; mix reproduction is byte-identical.
- Canonical source lint/runtime/layout checks report zero errors and zero warnings. Nine layout samples and 61/61 text contrast checks pass; four informational text overlaps occur in intentional crossfades. The checker's automated motion module was disabled and is not claimed as passed.
- Source and capture hashes were checked. Python source parses. Existing demo parity passes seven scenarios, six built-in analyses and 18 altered-analysis blocks; maximum numerical discrepancy is 3.70e-16. No app or statistical result was changed for this film.
- The watch page includes native controls, download, text version and scientific limits. At a 390-pixel viewport its document width is 390 pixels; the text disclosure opens.

## Scientific provenance

The edit uses genuine local v0.1.1 application states from app commit `079eb1c4db042580d60bd28f0708df068109d21e`, reframed for legibility. It is an edited sequence, not a continuous UI recording. Seed 42, scenario `fork`, three `a-low` actions, n=8 and cost=2 each reproduce P(A→Y)=0.03248648669137345 from a 0.5 prior. Initial expected query gain is 0.2890013991841607 nats. The unresolved B/Z orientation and finite candidate-family limitation remain visible. Withdrawing exchangeability in the measured-confounding example blocks the estimate. An accepted assumption is not proof. No real-world causal verification or new superiority claim is made.

## Review scope

The user selected “Keep this cut.” The framing correction retains the selected content, timing and soundtrack. The agent inspected current-source previews and extracted encoded frames and measured audio levels/sync. It has no listening feed or independent continuous audiovisual perception; the user selection is not represented as a full continuous listening certification.

The matching release source ZIP excludes the MP4, raw renders, caches, local paths in logs and third-party vendor bytes. The public app tag and original release assets remain separate from this media update.
