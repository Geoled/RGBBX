# soundgen — engine sound synthesizer

An offline **and** real-time engine-sound synthesizer for the Roblox engine
simulator. It turns a preset (engine geometry, exhaust, intake, mechanical,
turbo, space, character) into a complete, mix-ready sound set:

* 5 seamless **loop layers** (idle / part throttle / WOT / overrun / rev limiter)
* up to 7 **events** (crank, catch, shutdown, backfire, blow-off, turbo spool, knock)
* a `manifest.json` and a drop-in `EngineSoundManifest.luau` for Roblox

The syntheziser does not use samples. Everything is generated from the physics
of combustion events, so a 4-cylinder, a cross-plane V8, a Wankel and a turbine
all come out of the same code with different numbers.

---

## Quick start

```bash
cd tools/soundgen
python -m soundgen list                                    # what presets exist
python -m soundgen info    --preset presets/v8_crossplane.toml
python -m soundgen selftest                                # sanity checks
python -m soundgen bench   --preset presets                # throughput
python -m soundgen render  --preset presets --out out      # render everything
python -m soundgen realtime --preset presets/i4_turbo.toml --interactive
python -m soundgen new     "My Engine"                     # blank preset to edit
```

Only **numpy** is required. `tomllib` (3.11+) is used for presets, and `wave`
for WAV writing — both are standard library. `sounddevice` is optional and only
needed for live speaker output.

---

## How the sound is made

### Firing frequency

Everything starts with how often cylinders fire:

```
f_fire = cylinders * rpm / (60 * cycle_revolutions)
```

`cycle_revolutions` is 2 for a four-stroke, 1 for a two-stroke and a Wankel
(one firing per rotor per shaft revolution). A 2.0 litre I4 at 3000 rpm fires
100 times a second; a 13B at 9000 rpm fires 300 times a second. That number is
the perceived pitch, and it is why the model gets the character of an engine
right for free.

### Signal chain

```
combustion pulse train ──► exhaust resonator bank ──► pipe comb ──► muffler  ┐
   (one pulse per        (body resonances, f ∝     (2L/c delay,   (low-pass) │
    firing event)         size^-0.3)                sum of echoes)           │
                                                                             │
intake resonator bank  +  intake broadband noise  (throttle dependent)       │
valvetrain / injector tick train                                             │
crank thump + piston slap (per firing) + friction bed                        ├─► sum
turbo whine + whoosh  /  accessory whine                                     │
                                                                             │
  ──► presence bell ──► air rolloff ──► DC block ──► soft clip ──► gain      ┘
        ──► outdoor convolution reverb (wet / dry mix)
```

* **Combustion pulse** — a fast pressure rise, exponential decay, a resonant
  component and a short noise burst (the initial "bark"). Smaller cylinders
  produce a faster, brighter pulse.
* **Exhaust resonators** — the pipeline's body resonances, modelled as the
  analytic impulse response of a bank of damped resonators.
* **Pipe comb** — `delay = 2 * pipe_length / speed_of_sound(exhaust_temp)`,
  summed over several echoes. This is what gives an exhaust its hollow
  "hollowness" and is the main character control per preset.
* **Muffler** — a low-pass at `muffler_cutoff_hz`.
* **Intake** — the same event train through an intake resonator bank, plus
  broadband noise whose level tracks throttle. The noise runs through the *same*
  resonator bank, so it is coloured by the intake rather than being a flat
  shelf, and `noise_tilt_db_oct` adds a falling slope (turbulent flow is never
  white). This is what you hear on the intake side and at part throttle.
* **Mechanical** — the event train through a short bright tick filter. Every
  valve lifts *and* seats once per cycle (`valves_per_cyl`), the lift quieter
  than the seat impact, and the injectors tick independently of the valvetrain so
  rotaries and two-strokes get them too. The click is a damped ring plus a short
  band-limited noise burst, which is what makes it a tick instead of a ping. Each
  cylinder gets a fixed timing scatter (`mech_scatter_ms`) and its own brightness
  offset, so the train neither stacks into one waveform nor cancels the exhaust
  pulse it rides on.
* **Turbo** — a whistle whose frequency rises with spool, plus band-limited
  whoosh, optionally with blow-off flutter.
* **Mechanical bottom end** (`[mechanical]`) — a crank / big-end thump and a
  piston slap a few degrees after TDC, where the firing load actually peaks
  (offsetting them is both physical and what stops them meeting the exhaust pulse
  in antiphase); the slap returns, weaker, at the opposite TDC for four-strokes.
  Plus a band-limited friction bed and a cam-chain whine at half crank speed
  folded into the accessory cluster. All of it rides the firing grid, so it is
  physically driven rather than random. Gated off for turbines; slap gated off
  for rotaries.
* **Space** (`[space]`) — a real convolution with a synthetic outdoor impulse
  response (`dsp.outdoor_ir`): sparse early reflections plus a short dark tail,
  unit energy, seeded so a preset renders identically every time. Applied as the
  last linear stage with an equal-power wet/dry mix. Because it is LTI, a
  rendered loop stays exactly periodic; `layers.warmup_samples` reserves the
  whole tail so the reverb is already ringing when the loop starts.

### Why it can run in real time

Every filter above is linear and time-invariant, and the resonances depend on
**geometry, not rpm** — only the excitation and the turbo whistle change with
speed. So the filters are baked once into FIR kernels and applied with an
overlap-add FFT convolver, which is fully vectorised numpy. Measured on the
development machine:

| sample rate | speed |
|-------------|-------|
| 48 kHz      | 4.5–7× real time (≈15–22 % of one core) |
| 96 kHz      | 1.4–2.3× real time |

Because real-time streaming and offline rendering pull from the *same*
`EngineVoice`, the rendered WAV and the live preview are bit-for-bit the same
signal — the only difference is sample rate and where the samples go.

> Roblox cannot synthesise PCM at runtime, so the shipped game plays the
> rendered loops and shifts playback speed. The real-time path exists for
> previewing, tuning and testing presets in Python.

---

## Loop quality

The layers are **exactly periodic**, not crossfaded. That matters because an
equal-power crossfade is only correct for uncorrelated material; applied to a
correlated engine tone it adds up to +3 dB and modulates the pitch at the seam.

To make a loop close exactly, every periodic element is aligned to the loop:

| element | how it is made periodic |
|---------|-------------------------|
| firing grid | the cycle period is quantised to a whole (even) number of samples, so event *j* always lands on sample `j*P + round(off*P)` |
| loop length | a whole number of quantised cycles |
| noise | read from a fixed buffer indexed modulo the loop length |
| oscillators | frequencies snapped so a whole number of cycles fits the loop |
| limiter gate | modulators snapped, and smoothed circularly |
| filters | warm-up discarded until longer than the longest filter memory |

The result is verified by `selftest`, which renders one loop plus a following
block and asserts they are identical. Typical measured seam error is about
**−164 dB below signal level** (float round-off), reverb tail included.

Only the two *looping events* (`turbo_spool`, `knock`) use a crossfade, because
they are not periodic by nature.

---

## Loudness

Levels are deliberately not normalised per file. One gain is calibrated per
preset across the five layer operating points, so the natural progression from
idle to wide-open throttle survives into the exported assets:

```
i4_gasoline:  idle -13.2 dB   mid -12.1 dB   wot -8.4 dB   coast -22.9 dB   limiter -10.6 dB
```

One-shot events are normalised to their own designed peak instead, because the
game's mixer decides how loud a backfire sits against the engine.

Every rendered record also carries integrated **LUFS** (when `pyloudnorm` is
installed), which makes it easy to compare presets without guessing from peak
or RMS alone. `render --dither` applies TPDF dither to one-shots only; loops are
left bit-exact so their periodicity is preserved.

---

## Presets

A preset is a single TOML file describing the whole voice. Start from a bundled
one and edit it; `soundgen info` shows the derived numbers. `[block] octane`
(95 RON by default) drives the ignition-delay knock model, so a low-octane tune
will pull timing sooner under boost.

For hand authoring there are three aids:
`presets/_template.toml` lists **every** field with a comment (and, being
`_`-prefixed, is never picked up as a preset), `docs/CREATING_ENGINES.md` is the
step-by-step guide with a tuning table, and `python -m soundgen new "<name>"`
writes a ready-to-edit file. Unknown keys are a hard error with a "did you mean"
suggestion rather than a silently ignored line, and out-of-range values are
rejected at load.

```toml
name = "I4 Gasoline 2.0 NA"
seed = 20050412          # makes per-cylinder scatter reproducible

[engine]
idle_rpm = 850.0
redline_rpm = 7200.0
starter_rpm = 260.0

[block]
cylinders = 4
layout = "inline"        # inline | v | boxer | rotary | radial | turbine
ignition = "spark"       # spark | compression | turbine
strokes = 4
bore_mm = 87.0
stroke_mm = 87.0
compression = 10.5
bank_angle_deg = 0.0
# firing_offsets = [...] # optional: uneven firing (V-twin, big-bang)

[exhaust]
pipe_length_m = 1.10
pipe_temp_c = 620.0
comb_gain = 0.55
muffler_cutoff_hz = 2600.0
bank_asymmetry = 0.0     # V engines: offset bank 1's pipe length -> burble
modes = [[95.0, 2.2, 1.00], [190.0, 3.0, 0.85], [410.0, 3.5, 0.65]]

[character]
brightness_db = 0.0
aggression = 1.35        # 1.0 = clean, higher = saturated
lumpiness = 0.0          # 0..1, idle lope from a hot cam
irregularity = 0.05      # per-cylinder scatter
idle_floor = 0.22

[mechanical]             # crank/piston bottom end; rides the firing grid
valvetrain_gain = 0.079  # valve-seat click; lift is quieter (valve_open_gain)
valves_per_cyl = 4       # 2 = eight-valve head; the count sets the tick rate
injector_gain = 0.025    # injector tick, independent of the valvetrain
mech_scatter_ms = 0.35   # fixed per-cylinder timing scatter
crank_gain = 0.10        # big-end/main-bearing thump, a few degrees after TDC
crank_hz = 105.0         # scaled up for bigger engines
slap_gain = 0.08         # piston slap; repeats weaker at the opposite TDC
slap_hz = 210.0
chain_gain = 0.035       # cam-chain tone at half crank speed
chain_hz = 340.0
friction_gain = 0.022    # continuous ring/bearing shear
friction_lo_hz = 55.0
friction_hi_hz = 380.0

[space]                  # outdoor convolution reverb (the last linear stage)
enabled = true
mix = 0.16               # wet fraction, equal-power
predelay_ms = 16.0
decay_s = 0.38           # time to -60 dB
damping_hz = 3800.0      # absorption of the tail
early_gain = 0.55
tail_gain = 0.72
seed = 1
```

Bundled presets:

| file | what it shows |
|------|---------------|
| `i4_gasoline.toml` | baseline NA four cylinder |
| `i4_turbo.toml` | turbo layer, blow-off, wastegate flutter |
| `v8_crossplane.toml` | two banks + pipe asymmetry → the V8 burble |
| `rotary_2rotor.toml` | Wankel: one firing per rotor per revolution |
| `i4_diesel.toml` | no throttle plate, injector clatter, low redline |
| `turbine.toml` | continuous combustion, no discrete firing |
| `lexus_lfa.toml` | V10 with a VE curve forced to make power at 8800 rpm |
| `peugeot_205_gti.toml` | small NA four tuned against real dyno figures |
| `peugeot_206_gti.toml` | same, 2.0 L 180 hp |

### Getting the V8 to burble

A cross-plane V8 fires evenly every 90°, so the burble does **not** come from
uneven firing. It comes from the two banks being acoustically different. Set
`bank_asymmetry = 0.18` and the two half-exhausts beat against each other.

---

## Using the output in Roblox

1. Upload the WAV files from `out/`.
2. Put the uploaded ids into `out/EngineSoundManifest.luau` (the `assetId`
   fields), or into your own mapping table.
3. Feed `loopStart` / `loopEnd` straight into `AudioPlayer.LoopRegion`
   (`NumberRange.new(loopStart, loopEnd)`) and set `Looping = true`.
4. Crossfade the layers by load and rpm, and set playback speed from
   `referenceRpm`:

```lua
-- pitch is relative to the rpm the loop was actually rendered at
player.PlaybackSpeed = math.clamp(currentRpm / referenceRpm, 0.5, 2.0)
```

Listen with headphones the first time: the difference between the idle and WOT
layers is more than 10 dB, and that is intentional.

---

## Commands

| command | purpose |
|---------|---------|
| `list` | list bundled presets |
| `info --preset …` | derived timing, displacement, layer/event plan |
| `render --preset … --out DIR` | render WAVs + manifests |
| `bench --preset …` | synthesis throughput, no audio hardware needed |
| `realtime --preset … [--interactive]` | stream to speakers (needs `sounddevice`; prefers WASAPI) |
| `new <name> [--out FILE]` | write a fresh preset from `_template.toml` |
| `selftest` | end-to-end checks, including loop periodicity |

Render options: `--sr`, `--bits 16|24`, `--channels 1|2`, `--block`,
`--layers-only`, `--events-only`, `--normalize`, `--dither`, `--quiet`.

With `--interactive`, the keyboard drives the engine live:
`W`/`S` throttle, `A`/`D` rpm, `Q` quit.

---

## Layout

```
tools/soundgen/
├─ soundgen/
│  ├─ dsp.py        kernels + streaming processors (convolution, noise, oscillators,
│  │                impact kernels, synthetic outdoor IR)
│  ├─ layout.py     geometry, firing order, bank assignment, size scaling
│  ├─ config.py     preset dataclasses + TOML load/save + `discover_presets`
│  ├─ voice.py      EngineVoice — the synthesiser itself, and gain calibration
│  ├─ layers.py     the 5 loop layers, limiter chopping, warm-up length
│  ├─ events.py     the one-shots
│  ├─ render.py     offline driver
│  ├─ realtime.py   live playback + benchmarking
│  ├─ manifest.py   JSON + Luau manifest generation
│  ├─ wavio.py      16/24-bit WAV via `soundfile`, stdlib `wave` fallback
│  └─ cli.py        command line
├─ presets/*.toml      the 23 engines
├─ presets/_template.toml   every field, annotated (never discovered as a preset)
└─ requirements.txt
```

---

## Notes and limits

* The model is a **synthesis** model, not a measured sample library. It gets the
  firing rate, the resonance structure and the load behaviour right; it does not
  reproduce a specific car.
* `aggression` is a real saturation stage, not a limiter. Above roughly 2.5 the
  output starts to square off.
* Turbo boost is estimated as
  `max_boost_bar * clamp((rpm - spool_rpm) / spool_span_rpm) * load`, which is
  what the simulator uses, so sound and physics agree.
* Render at 96 kHz if you want maximum quality for an offline master; the same
  presets are used unchanged.
