# CAUSALOS launch films

The current [v0.1.1 film and editable sources](v0.1.1/README.md) show the redesigned interface and assumption gate. [Watch the current film](https://nazeeh111.github.io/CausalOS/launch/). The original v0.1.0 film is preserved below.

## Original v0.1.0 film

`brag.mp4` is the final 20-second landscape film. `brag.jpg` is the selected settled poster and is also baked into frame zero. The middle scenes use genuine captures of the functioning CAUSALOS laboratory. Exact numerical annotations derive from `benchmarks/flagship_replay.json`, seed 42, before and after three `a-low` interventions. Results are explicitly simulated and conditional on the supplied finite model family. The B/Z nuisance orientation is unresolved.

The authored HTML, storyboard, identity, audio generator and all local media remain editable. `generate_audio.py` creates the original deterministic score, with no borrowed recording or recognizable melody. Audio and authored film source are covered by the project MIT license. There is no voice narration.

## Re-render

Install Python with NumPy, Node.js 22+, Chrome, FFmpeg and HyperFrames 0.8.33 or a compatible newer local installation. Fetch GSAP 3.14.2 with `python setup_composition.py`; this verifies a pinned checksum. GSAP is a separate no-charge third-party dependency under the [GSAP standard license](https://gsap.com/standard-license/) and is excluded from this source distribution. The core CAUSALOS app does not depend on it. The Iowan Old Style font uses an installed macOS system face; Palatino or Georgia are fallbacks on other platforms.

Run `python generate_audio.py` to regenerate the score. From `composition/`, use `hyperframes lint`, `hyperframes check --samples 9` and `hyperframes render --output ../brag-new.mp4 --fps 30 --quality standard --workers 1`. HyperFrames versions may use `inspect` in place of `check`; consult the installed CLI. The development render used the already installed pinned local runner, documented in `verification.md`.

Preserve the last valid film while rendering revisions. After successful rendering and frame inspection, select the best settled working-app frame and replace only frame zero with that poster; retain all other frame timing and audio. See `verification.md` for the delivered file's measured properties and verification scope.
