# CAUSALOS v0.1.1 launch film

The selected master is `causalos-v0.1.1-launch.mp4`, with a separate settled-frame poster. The 24-second landscape edit shows the actual v0.1.1 interface and its assumption gate. It is an edited sequence of genuine UI states, not a continuous screen recording. The previous `../brag.mp4` film remains preserved as the v0.1.0 version.

Watch at [the public film page](https://nazeeh111.github.io/CausalOS/launch/) or download it from the [v0.1.1 release](https://github.com/nazeeh111/CausalOS/releases/tag/v0.1.1).

## Evidence and sources

`evidence.json` pins the application commit, exact replay values, source hashes and capture hashes. `capture-crops.json` records the lossless viewport crop rectangles. The full native viewport PNGs are retained beside the used crops. Explanatory annotations stay outside the app images. No graph, control, estimate or numerical value is painted into the application UI.

The sequence uses seed 42, scenario `fork`, three `a-low` interventions, eight samples per intervention and six total cost units. The query probability goes from 0.5 to 0.03248648669137345; the film displays 50.0%→3.2%, matching the application. The B/Z nuisance orientation remains unresolved. The gate comes from the real local `confounded` example, seed 42, with exchangeability withdrawn after a valid analysis.

`composition/index.html`, `DESIGN.md`, `storyboard.md`, screenshots and audio sources remain editable. `generate_audio.py` synthesizes the original 24-second score; `mix_audio.py` reproduces its two-pass mix with installed FFmpeg. No sampled recording or narration is used.

## Reproduce

Use installed Python with NumPy, Node.js 22+, Chrome, FFmpeg and HyperFrames 0.8.33. The authored video has a 1920×1080 canvas at 30 fps. The source uses installed Iowan Old Style, with Palatino/Georgia fallbacks; other fonts can change line wrapping. It has no model API requirement.

1. Run `python setup_composition.py` to fetch the separately licensed GSAP 3.14.2 runtime. The script verifies its pinned SHA256 and refuses to replace differing bytes. Vendor bytes are excluded from the editable source archive. GSAP retains its [standard license](https://gsap.com/standard-license/); it is not a CAUSALOS app dependency.
2. Regenerate the original track with `python generate_audio.py`. For a new mix, use `python mix_audio.py --output composition/assets/score-mix-new.wav` and update the audio source in HTML. The mixer preserves existing outputs.
3. On the production Mac, use the installed pinned runner `/Users/nazeeh/.local/share/codex-tools/hyperframes-0.8.33-local/local.py` to run `lint` and `check --samples 9` against the absolute `composition/` path. The check targets the canonical 1080p `index.html`.
4. Render with `python render_local_screenshot.py render /absolute/path/to/composition --composition index-overscan.html --output /fresh/path/render-overscan.mp4 --fps 30 --quality high --workers 1`. The wrapper retains the installed runner's checksum, isolated environment and output-preservation guards. It requires that pinned runner on the production Mac; the composition is portable to an existing compatible HyperFrames installation.
5. Run `python finish_export.py /fresh/path/render-overscan.mp4 --output /fresh/path/causalos-v0.1.1-launch.mp4`. The production entry adds a 200-pixel capture margin below the authored 1920×1080 scene because the installed renderer clipped lower captions in an ordinary 1080p export. The finalizer removes this margin using H.264 picture-crop metadata and stream-copies both video and audio, with no second encoding. A second stream-copy pass refreshes the MP4 track/sample dimensions from the cropped stream, so browsers report 1920×1080 before decoding the first frame. Both commands require fresh outputs and preserve the previous master.
6. Verify duration, 720 frames, dimensions, codecs, audio and full decode with FFprobe/FFmpeg. Inspect actual encoded scene and transition frames, including lower captions. The finalizer was reproduced byte-for-byte for this master. See `verification.md` and `media-verification.json` for measured properties and review limits.

The source and original audio are MIT licensed. Third-party dependencies retain their own licenses. The film adds no scientific superiority claim.

The editable-source ZIP omits the MP4 itself; download the matching master separately from the same release.
