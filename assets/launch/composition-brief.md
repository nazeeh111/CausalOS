# HyperFrames brief: CAUSALOS LAB

Create a 20-second 1920 × 1080 launch film in `composition/`, exported as `brag.mp4`. Follow `brag-plan.md` and `DESIGN.md`.

Read the current `causalos/web/index.html`, `styles.css`, and `app.js`. Product copy to preserve: “Same observations. Different explanations.” and “Evidence before conclusions.” The actual experiment lab before and after a simulated intervention is the centerpiece. Use the main agent's current functioning-app screenshots and the executed `benchmarks/flagship_replay.json`; retain their provenance. Do not invent an estimate, probability, comparison or supported causal claim.

The four scenes are ambiguity (0–4.5), actual experiment selection (4.5–9), actual posterior update (9–15), identity and repository (15–20). Use crossfades and individual entrances. The explicit simulation annotation persists over intervention results. Keep labels at least 20px and statement lines above 60px. Screenshot details must remain legible in the final render.

Use the installed local HyperFrames runner with its pinned runtime. Use only local media. Copy the installed GSAP dependency for rendering, but exclude vendor bytes from public source and supply a checksum-verifying setup script. Keep all authored source, original score generator and media. Use the generated score `assets/causalos-score.wav` on its own track at a restrained volume. Voice is disabled. Cue times are authored 4.5, 9 and 15 seconds; no reading time is sacrificed to a beat.

Before delivery run lint and current-source nine-sample checks, render with one worker, inspect representative extracted frames, verify duration/codecs and decode the complete film. Select a settled working-app poster frame, save `brag.jpg`, and bake its pixels only into frame zero while preserving duration and audio. Write `share-copy.txt` and `verification.md`.
