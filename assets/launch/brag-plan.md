# CAUSALOS: evidence before conclusions

## Planning rubric
1. **App:** a local causal-analysis and experimental-design laboratory that makes assumptions, competing graphs and simulated evidence visible.
2. **Strongest line:** “Same observations. Different explanations.”
3. **Visual hook:** two cobalt causal arrows point in opposite directions between the same observed variables.
4. **Actual UI:** the experiment laboratory before intervention and after its measured posterior update.
5. **Shortest satisfying film:** 20 seconds, enough to read the problem and see an experiment change the evidence.
6. **Tone:** polished; an editorial research instrument, with calm, deliberate motion.
7. **Audio:** original synthesized quiet electronic score and restrained interaction accents. No narration. Bundled music is omitted because its redistribution license is not established.
8. **Share copy:** “CAUSALOS LAB makes causal assumptions visible and lets competing explanations face a simulated experiment. Open source, CPU compatible, and reproducible.”
9. **User flow:** inspect competing explanations → choose an informative intervention → reveal the simulated observation and update model probabilities.

## Angle
The problem is observational ambiguity. The centerpiece is the functioning laboratory. The result is a model-conditional update, never proof of a real-world causal direction. Exact numerical labels must come from the executed flagship replay JSON.

## Visual identity
Palette and typography follow DESIGN.md and the current laboratory source: paper `#f5f4ee`, ink `#222a31`, cobalt `#2846c9`, amber `#936628`, Iowan Old Style statements, system interface text, SFMono labels.

## Storyboard: 20 seconds, 1920 × 1080 landscape
### 1. Observational ambiguity · 0–4.5s
The source line “Same observations. Different explanations.” enters above two opposing A/Y graph hypotheses. A footer labels these as hypotheses with equivalent observational distributions. The statements hold for more than 3 seconds. The first quiet score tone lands with the graph.

### 2. Choose an experiment · 4.5–9s
Crossfade into a genuine current-source capture of the working experiment UI before intervention. A small label reads “Choose the next experiment.” A subtle camera move directs attention to the action panel. A generated click accent accompanies the visible experiment handoff. Screenshot content remains unchanged.

### 3. Reveal the evidence · 9–15s
Crossfade to the same functioning UI after the selected simulated intervention. The exact selected action and model-conditional probability are readable in a short annotation. The persistent label “SIMULATED INTERVENTION” prevents confusion with real-world evidence. Camera settles and holds. Posterior detail is shown from execution, never fabricated or smoothed.

### 4. Evidence before conclusions · 15–20s
Crossfade to the CAUSALOS LAB identity and source tagline “Evidence before conclusions.” One restrained line reads “Open source · Local research · Reproducible.” Repository address appears as a location for the released source, subject to release verification by the main agent. The score resolves and fades.

## Audio treatment and cue guidance
The original 20-second score is generated from deterministic sine partials by `generate_audio.py`, which is released with the source. It uses gentle harmonic tones, not a recognizable borrowed melody. Strong events at 4.5, 9 and 15 seconds align with scene handoffs. Quiet interface accents occur at 8.8 and 9.0 seconds. The source creates its own cue metadata. Bass/RMS energy may subtly affect the logo's presence; no spectrum graphics are added. Readability controls timing.

## Evidence and publication controls
Use real screenshots supplied after application QA and the exact flagship replay output. No private data appears. Keep GSAP vendor bytes local and ignored; the setup script documents the no-charge third-party license and verifies downloaded bytes. User authorization covers rendering and public release; no additional preview confirmation is required.
