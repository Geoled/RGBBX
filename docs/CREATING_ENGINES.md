# Creating an engine

Every engine in the simulator is one TOML file. There is no editor and no
build step: copy the template, change the numbers, press play. The same file
drives the sound, the physics, the dyno and the 3D cutaway, so all four always
agree.

> **По-русски, коротко.** Скопируйте `tools/soundgen/presets/_template.toml` в
> `tools/soundgen/presets/my_engine.toml` (или выполните
> `python -m soundgen new "My Engine"`), поменяйте `name`, геометрию и
> `[character]`, затем запустите `run.bat --preset my_engine`. Файлы, имя
> которых начинается с `_`, движками не считаются. Все поля необязательны:
> что не указано, берётся из шаблона. Опечатки в именах ключей вызывают
> понятную ошибку с подсказкой.

---

## 1. Quick start

```bat
cd tools\soundgen
python -m soundgen new "My Engine 2.0"      :: writes presets\my_engine_2_0.toml
python -m soundgen info  --preset presets\my_engine_2_0.toml
python -m soundgen render --preset presets\my_engine_2_0.toml --out out
cd ..\..
run.bat --preset my_engine_2_0
```

Or simply copy the template by hand:

```
tools/soundgen/presets/_template.toml  ->  tools/soundgen/presets/my_engine.toml
```

The launcher, the app's engine list, `--list` and the montage all discover
`presets/*.toml` automatically and ignore names starting with `_`.

Check the file at any time:

```bat
python -m simulator --list
```

A broken preset is reported with the file name, the offending key and a
suggestion, and it never takes the rest of the roster down with it:

```
my_engine.toml: [character] unknown option 'brigthness_db' (did you mean 'brightness_db'?)
```

---

## 2. What the sections do

| Section | Controls |
| --- | --- |
| `[engine]` | idle, redline, starter speed |
| `[block]` | cylinder count, layout, bore/stroke, compression, octane, firing, VE curve |
| `[exhaust]` | pipe length/temperature, reflections, muffling, resonance bed, flow noise |
| `[intake]` | induction roar, intake pulses and their resonances |
| `[turbo]` | boost, spool, whistle, whoosh, blow-off |
| `[turbine]` | air-standard Brayton cycle for `layout = "turbine"` |
| `[mechanical]` | valvetrain, injectors, crank/bearings, pistons, chain, accessories |
| `[space]` | the outdoor convolution reverb |
| `[character]` | overall brightness, aggression, irregularity, idle behaviour |
| `[render]` | offline/stream settings; leave at the defaults |

### `[block]`: geometry is the biggest lever

Displacement, compression and the VE curve decide the dyno numbers. Bore and
stroke decide the *character*: a long-stroke engine (stroke > bore) has a
lower, gruffer note; an oversquare engine revs and sounds harder.

`octane` is the petrol RON used by the ignition-delay knock model. It defaults
to 95; lower it to 91-93 for a car on bad pump fuel, or raise it to 100-105 for
a race engine. Diesel and turbine presets ignore it. Compression still matters
more: a high-compression engine on low-octane fuel will pull timing sooner.

`layout` changes the firing pattern and the 3D model:

| `layout` | Meaning |
| --- | --- |
| `inline` | I3/I4/I5/I6, one exhaust bank |
| `v` | V6/V8/V10/V12, two banks (odd/even cylinders) |
| `boxer` | flat-4/flat-6, two banks |
| `rotary` | Wankel; one firing per rotor per shaft revolution |
| `radial` | aircraft radial; odd firing is normal |
| `turbine` | gas turbine; no discrete firing events at all |

`firing_offsets` is optional. Leave it empty for even firing. A cross-plane V8
uses an uneven pattern, which is what produces the burble:

```toml
firing_offsets = [0.000, 0.875, 0.375, 0.250, 0.625, 0.500, 0.750, 0.125]
```

### The volumetric-efficiency curve

This is the one part of the physics a preset may override, because it is what
separates a truck engine from a race engine:

```toml
ve_peak_rpm_frac = 0.0   # 0 = auto (0.55): peak VE at 55 % of the redline
ve_droop = 0.0           # 0 = auto (0.30): fraction of VE lost by the redline
ve_peak_value = 0.0      # 0 = auto (~0.98); >1.0 models ram-effect breathing
```

A 9000 rpm V10 that holds its torque to the limiter needs something like
`ve_peak_rpm_frac = 0.70`, `ve_droop = 0.06`, `ve_peak_value = 1.15`. Leave the
zeros for an ordinary engine.

### `[exhaust]`: where most of the voice lives

* `pipe_length_m` sets the quarter-wave line. Longer = lower and boomy; shorter
  = harder and more metallic.
* `pipe_temp_c` raises the formants as it rises (hot gas is faster).
* `muffler_cutoff_hz` is the single most effective "how loud is it" control.
  A stock system is 2600-4200 Hz; a race system is 5500-7000 Hz.
* `modes` is the body-resonance bed, `[hz, Q, gain]`. Keep the gains falling
  with frequency. **Q should rise with frequency**; a flat Q bank sounds
  synthetic. The narrow standing-wave lines come from the comb, not from here.
* `bank_asymmetry` (0.05-0.2) makes a V engine's two banks slightly different,
  which is what turns a clean tone into a burble.

### `[intake]`: induction roar

* `noise_gain` is the loudness of the broadband induction noise, `pulse_gain`
  that of the discrete intake pulses. Both are shaped by `modes`.
* `band_lo_hz`/`band_hi_hz` bracket the noise before `modes` colours it.
* `noise_tilt_hz`/`noise_tilt_db_oct` set the slope of that noise. Air past a
  throttle plate is **turbulent, not white**: at `0` dB/oct the band is a flat
  shelf and reads as tape hiss laid over the engine. `-2` (the default) gives a
  falling, roar-like slope; `-4` is darker still, `+2` brighter. The `modes`
  bank adds its own fall on top, so this is a trim rather than the whole shape.
* `runner_length_m` sets the quarter-wave timbre of the pulses, `runner_spread`
  detunes individual runners so they do not sum into one clean tone.

### `[mechanical]`: the part that makes it a machine

The defaults already include valvetrain ticks, injectors, crank/big-end
thumps, piston slap, chain whine, bearing shear and the accessory whine. Raise
`crank_gain`/`slap_gain` for an old, loose, big-bore engine; lower them for a
modern, tight one. Keep the frequencies low: these layers add weight, and a
bright mechanical layer reads as hiss.

The layer is physical rather than decorative. Every valve both *lifts* and
*seats* once per cycle, so `valves_per_cyl` (4 for a modern four-valve head, 2
for an eight-valve) sets the tick count; `valve_open_gain` makes the lift
quieter than the seat impact, as it is in life. `injector_gain` is independent
of the valvetrain, so a rotary or a two-stroke gets injector ticks without
valves. The crank thump and the piston slap land a few degrees after TDC,
where the load actually peaks, and `mech_scatter_ms` gives each cylinder a
fixed timing offset so the clicks neither stack into one waveform nor cancel
the exhaust pulse they ride on.

`valvetrain_bright_hz` is the metallic ring of the head; 3200-3600 Hz suits an
old iron head, 3800-4200 Hz a modern alloy one. Because the click carries a
short noise burst as well as the ring, raising the gain gives a sharper tick,
not more hiss.

### `[space]`: outdoors

The reverb is a real convolution with a synthetic outdoor impulse response:
sparse early reflections plus a short, dark tail. `mix` is the wet fraction.

| `mix` | Scene |
| --- | --- |
| 0.10-0.14 | open track, very little around |
| 0.15-0.19 | default: a car in the open, some ground bounce |
| 0.20-0.26 | narrow street, walls close by |

`decay_s` is time to -60 dB. Outdoors rarely exceeds 0.5 s. `damping_hz`
controls how dark the tail is; `early_gain` the discrete slap, `tail_gain` the
diffuse part.

### `[character]`: voicing, not physics

* `brightness_db` is a **bell** at `presence_hz`, not a shelf. +3 to +5 adds
  rasp without lifting hiss.
* `aggression` saturates the mix. 1.0 is clean, 1.4-1.6 is a race engine.
* `irregularity` scatters cylinder amplitudes; `ccv` scatters each firing.
  Together they stop a loop from sounding like a test tone.
* `lumpiness` gives a hot-cam idle lope.
* `pulse_ref_rpm` / `pulse_min_scale` control how the combustion pulse shortens
  with rpm. Leave them alone unless you are modelling a very high-revving
  engine.

---

## 3. Recipes

### Naturally aspirated inline-four (the reference)

Use the template unchanged. Then:

* more top-end: `ve_peak_rpm_frac = 0.62`, `ve_droop = 0.18`;
* more rasp: `brightness_db = 4.5`, `presence_hz = 1600`;
* louder: `muffler_cutoff_hz = 5200`.

### Cross-plane V8

```toml
[block]
cylinders = 8
layout = "v"
bank_angle_deg = 90.0
bore_mm = 101.6
stroke_mm = 92.0
compression = 10.5
firing_offsets = [0.000, 0.875, 0.375, 0.250, 0.625, 0.500, 0.750, 0.125]

[exhaust]
bank_asymmetry = 0.13
pipe_length_m = 1.15
muffler_cutoff_hz = 3200.0

[character]
lumpiness = 0.35
irregularity = 0.07
```

### Turbo four

```toml
[turbo]
enabled = true
kind = "turbo"
max_boost_bar = 1.4
spool_rpm = 1500.0
spool_span_rpm = 1400.0
whistle_gain = 0.22
flutter = true

[block]
compression = 9.0
octane = 98.0          # a bit of margin under boost

[exhaust]
pipe_temp_c = 720.0
muffler_cutoff_hz = 3400.0
```

`max_boost_bar` also drives the physics: the dyno will show the boost, the
manifold pressure and the extra torque. `kind = "supercharger"` makes it spool
instantly.

### Gas turbine

A turbine is not a reciprocating engine, so it uses a separate mean-value
Brayton cycle. `[turbo]` still controls the *sound* (whine and whoosh), while
`[turbine]` controls the shaft:

```toml
[block]
layout = "turbine"
ignition = "turbine"
compression = 12.0       # display/reference value; cycle PR is below

[turbine]
pressure_ratio = 12.0          # compressor pressure ratio at the redline
idle_turbine_inlet_c = 650.0   # firing temperature at idle
max_turbine_inlet_c = 1200.0   # firing temperature at full power
compressor_efficiency = 0.80
turbine_efficiency = 0.88
combustor_efficiency = 0.98
mechanical_efficiency = 0.97
mass_flow_kg_s = 15.0
pressure_loss = 0.95
spool_tau_s = 1.2
wind_down_tau_s = 2.0
```

The model computes compressor and turbine work from the pressure ratio,
firing temperature and component efficiencies, subtracts the compressor's
share, and converts the remaining shaft work into torque. Raising
`max_turbine_inlet_c` or `pressure_ratio` makes more power; raising
`mass_flow_kg_s` scales the whole machine. `spool_tau_s` is the acceleration
time constant, so a heavy industrial rotor should be slower than a small
turboshaft. `[turbo] spool_rpm` and its span still shape the audible whine; they
do not change the thermodynamic cycle.

### Rotary

```toml
[block]
cylinders = 2          # rotors
layout = "rotary"
bore_mm = 102.0
stroke_mm = 80.0

[mechanical]
valvetrain_gain = 0.0  # rotary: no poppet valves
injector_gain = 0.045

[character]
brightness_db = 4.2
presence_hz = 1800.0
```

### Two-stroke

```toml
[block]
cylinders = 1
strokes = 2
layout = "inline"
```

A two-stroke fires every revolution, so the firing frequency doubles for the
same rpm. Keep `pipe_length_m` short (0.5-0.8) for the expansion-chamber
character and raise `brightness_db`.

---

## 4. Tuning by ear

| Symptom | What to change |
| --- | --- |
| Too hissy / fizzy | lower `intake.noise_gain`, make `intake.noise_tilt_db_oct` more negative, lower `brightness_db`, lower `hiss_gain`, lower `air_cutoff_hz` |
| Too thin / no body | raise `comb_gain`, add a low `modes` entry, lower `muffler_cutoff_hz` |
| Too boomy / one-note | raise `muffler_cutoff_hz`, add modes between 300-1200 Hz |
| Wrong pitch | the firing frequency is set by cylinders/layout/rpm, not by modes. Check `python -m soundgen info` |
| A pure whistle on top | lower `whistle_gain` / `accessory_gain`, or lower `idle_whistle` |
| Sounds synthetic | raise `irregularity`, `ccv`, `bank_asymmetry`; stagger `runner_spread` |
| Too much room | lower `[space].mix` or `decay_s` |
| Mechanical noise too clean | raise `crank_gain`, `slap_gain`, `chain_gain` |

The `selftest` and `bench` commands tell you whether a preset is valid and fast
enough without opening the app:

```bat
cd tools\soundgen
python -m soundgen info --preset presets\my_engine.toml
python -m soundgen bench --preset presets\my_engine.toml --block 512
```

---

## 5. Rules the renderer relies on (for the curious)

* A loop is rendered from steady state and cut to a whole number of engine
  cycles, so every periodic layer and noise source has to be exactly periodic.
  The built-in generators already do this; a new *field* only needs care if it
  introduces a new continuous noise source (use `CircularNoise` in loop mode)
  or a free-running oscillator (snap its frequency to the loop).
* The reverb is linear and time-invariant, so it preserves loop periodicity once
  its impulse response has fully warmed up. `warmup_samples` accounts for it.
* All filters are FIR kernels or fractional delay lines; there is no scipy.
