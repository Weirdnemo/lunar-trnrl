# Research Log: RL guidance for lunar powered descent with terrain-relative navigation (TRN)

Working title: *Uncertainty-aware learned descent guidance under degraded terrain-relative navigation.*
Owner: Nimesh Chauhan. Log kept by Claude alongside development; **append, don't rewrite**. Every number below was produced by code in this repo (`scripts/`, raw per-episode data in `results/*.json`), except E019 and E020, which were run by the user on their machine on their TIFFs; for those I have the printed summaries only (the JSON files are on the user's machine). Where something is an assumption or a limitation it says so.

---

## 0. Status at a glance (update this block every session)

| | |
|---|---|
| Last updated | 2026-10-15 (through E029) |
| Stage | Simulator, EKF+TRN, baseline, tests (20 passing), benchmark, two classical improvements (decoupled altimeter, covariance-triggered hold), PPO on synthetic terrain (E014, E018) and on random windows of real PGDA terrain (E019, E020), explainer figures + page (E015), real-tile runs (E016), local 3D sandbox (E017). |
| Terrain | Synthetic (fractal + craters) for the main benchmark. **Real:** two 8 km PGDA LOLA tiles, Site 04 (Shackleton rim) and Site 23 (Malapert massif), 5 m block-averaged to 10 m (E016, E019, E020). Both are rugged (relief std 270 to 310 m), about 90% interpolated, and neither is a flat low-relief site. Haworth (29.8 km, E021) added with a proper 70/30 geographic split. |
| Headline findings | (1) Landing accuracy is limited by *navigation*, not guidance: precise landing is 91 to 98% when TRN gives at least one fix and 6 to 17% when it gives none (E012). (2) All headroom is on low-relief terrain (E012). (3) A hand-written covariance-triggered hold + decoupled altimeter lifts precise landing from 75.5% to 89.0% on `mixed_hard` (E011, E013). (4) **First PPO residual runs (1 seed, 600k steps) did not beat the baseline or the hold rule** (E014); H1 and H2 not supported by this evidence. (5) **Retraining on the decoupled filter (E018) and training on random real windows then testing on a different real site (E020) also gave no RL gain**; late training degrades the final model (final cov 85.5% vs baseline 97.5% on the held-out site). On rugged real terrain the baseline already lands 95 to 98% precisely and TRN nearly always fixes, so there is little headroom (E016, E020). All results are one seed per variant. (6) **Three real sites (Site 04, Site 23, Haworth) all give baseline precise landing of 95 to 99.5% (E016, E020, E021); the navigation-limited regime of the synthetic benchmark does not appear on them.** (7) **The 300 m big_init losses of E022 were the 20-cell TRN search window (E023). With the window widened to 3 sigma, site04 stays at 99% up to 1000 m initial error; site23 drops to 73 to 80% at 600 to 1000 m, with ambiguity-type failures. Real headroom only appears on the rugged-and-ambiguous tile at large initial error.** (8) **With perfect navigation the baseline reaches only 96 to 99% on real tiles (unsafe designated targets, E027); a larger terrain-matching patch lifts the fixed baseline at 600 m error from 78% to 90% (E029).** (9) **A learned hold/detour decision, warm-started and trained on 2200 episodes x 2 seeds, equals always-hover (87.5%) and does not beat the hand-written hold (89.0%) on the synthetic benchmark (E029).** |
| Hypotheses so far | H1 (covariance helps): not supported (E014, E018, E020; cov was worse in all three). H2 (RL beats fixed guidance on p90): not supported (E014, E018, E020). H3 (flies over textured terrain): not tested. One seed each, so "not supported" is not "refuted". |
| Next | (1) A real-tile decision setting once enough decision episodes exist; (2) test the mechanism behind 'detour away from the target' (time bought vs second viewing geometry); (3) [done E038] more Site 23 episodes; (3b) cause of sigma1000 > sigma600; (4) justify the field-of-view, offset and map-noise values; (5) 3 to 5 independent training seeds for learned results; (6) hazard retargeting is the only route above the ~98% ceiling. |

---

## 1. Problem statement and hypotheses

**Problem.** A lander flies a powered descent from ~1 to 1.4 km altitude to a designated site on the Moon. It does not know where it is: the inertial solution drifts, and the only absolute position information is terrain matching (lidar height patch vs. an onboard map). Terrain matching works well on rough terrain and fails on smooth plains. Can a learned policy that is *shown the navigation uncertainty* land more accurately than classical guidance that ignores it?

**Hypotheses (pre-registered before any RL run, so results can be judged against them):**

- **H1.** Adding navigation covariance and fix-age to the policy observation improves precise-landing rate over an otherwise identical policy without them, in degraded-TRN scenarios.
- **H2.** A learned policy beats fixed-target polynomial guidance on p90 landing error when TRN is intermittent (low-texture terrain or sensor blackout).
- **H3 (exploratory).** The policy learns to prefer trajectories over textured terrain (observability-aware flight). Evidence would be higher TRN fix rate for the policy than for the baseline at equal fuel.

**Null/negative results are publishable here**: if the baseline stays competitive, that is itself a finding about when learning is worth it.

---

## 2. System description (as built)

Code: `trnrl/terrain.py`, `nav.py`, `env.py`, `guidance.py`. Frame: local ENU, metres, +z up.

- **Dynamics.** 3-DoF point mass, lunar gravity 1.62 m/s², m0 = 2000 kg, dry 1500 kg, thrust 1200 to 6000 N (throttle floor 20%), Isp 311 s, thrust-vector tilt limit 40 deg. Integrated at 4 substeps per 1 s control step; touchdown detected by interpolating the ground crossing.
- **Truth vs. onboard map.** Truth terrain is 10 m/cell. The onboard map is the truth Gaussian-filtered and decimated by 2 (20 m/cell). The mismatch (0.3 to 0.5 m RMS on the current synthetic terrain) is the map error the filter must tolerate.
- **Navigation.** 9-state EKF (position, velocity, accelerometer bias). Sensors: IMU (noise + bias random walk), Doppler velocimeter, terrain-aided radar altimeter, and TRN every 2 s between 150 and 2500 m AGL. TRN = normalised cross-correlation of a 15x15 lidar patch (280 m wide) against the onboard map within a 3-sigma search window (max +/-20 cells), sub-cell peak interpolation, then gating:
  - reject if patch relief < 0.3 m ("flat"),
  - reject if best NCC < 0.7,
  - reject if a competing peak outside the main correlation lobe is within 0.03 NCC of the best (ambiguity),
  - chi-square innovation gate (99.9%).
  Fix covariance comes from a Cramer-Rao-style bound on map gradients over the patch (inflated x2, floor 0.4 cell), so low-gradient terrain gives weak fixes automatically.
  Initial knowledge error: 120 m horizontal, 30 m vertical, 0.3 m/s velocity (1-sigma).
- **Environment (Gymnasium).** Action: throttle + 2D thrust-direction command (3 values in [-1, 1]). Observation (222-d): relative position/velocity from the *estimate*, estimated AGL, fuel fraction, log horizontal and vertical sigma, time since last fix, plus crops of the onboard map (9x9 hazard-safe mask at two strides, 7x7 texture map). Truth is never in the observation. Reward: potential shaping toward target, fuel and time costs, terminal bonus/penalty.
- **Success definitions.** *Safe landing*: touchdown vertical speed <= 2 m/s, horizontal <= 1 m/s, on truth terrain with slope <= 10 deg and roughness <= 0.6 m. *Precise landing*: safe **and** within 50 m of the designated target. Precise landing is the headline metric (see E006 for why).
- **Baseline guidance.** Two-phase: polynomial (E-guidance style) to a 30 m gate above the designated target, then constant-rate vertical descent with PD horizontal hold. Uses the nav estimate only.

---

## 3. Experiment registry

All runs: 100 episodes, seeds 10000 to 10099, 8 synthetic sites, same scenario generator unless noted. Raw data: `results/<tag>.json`. Command: `python scripts/evaluate.py baseline 100 <tag>`.

| Tag | Condition | Safe landing | Precise (<=50 m) | Median / p90 landing err | Median fuel | TRN fix rate | Zero-fix episodes | Median NEES (dof=9) |
|---|---|---|---|---|---|---|---|---|
| `baseline_synthetic_trn` | nominal | 99% | **97%** | 1.1 m / 1.8 m | 148 kg | 92% | 0 | 6.7 |
| `baseline_no_trn` | TRN disabled | 93% | **42%** | 69 m / 336 m | n/r | 0% | 100 | 69.4 |
| `baseline_low_relief` | relief 4 m RMS, 8 craters, TRN on | 98% | **74%** | 4.2 m / 204 m | n/r | 21% | 23 | 8.8 |
| `baseline_mixed_hard` | 50% rough / 50% low-relief sites, 40% of episodes have a 30 to 70 s TRN blackout, **200 episodes** | 94% | **75.5%** | 1.9 m / 168 m | 152 kg | 53% | 42 | 8.1 |
| `baseline_mixed_hard_dec200` | same, decoupled altimeter | 99% | **83.0%** | 1.5 m / 110 m | 152 kg | 58% | 32 | 7.4 |
| `ppo_cov_s0_test` | PPO residual, with covariance obs, seed 0, coupled filter | 95% | **70.0%** | 4.4 m / 230 m | 119 kg | 51% | 52 | 8.3 |
| `ppo_nocov_s0_test` | PPO residual, covariance zeroed, seed 0, coupled filter | 98.5% | **78.0%** | 8.2 m / 218 m | 147 kg | 54% | 47 | 8.1 |
| `wait70_test_mixed_hard` | covariance-triggered hold (70 s), coupled filter | 98.0% | **80.5%** | 2.1 m / 246 m | 220 kg | n/r | 32 | 8.3 |
| `wait70_test_mixed_hard+dec` | covariance-triggered hold (70 s), decoupled filter | 99.5% | **89.0%** | 1.7 m / 60 m | 220 kg | n/r | 21 | 8.0 |

All `mixed_hard` rows: test split (held-out sites), 200 episodes, seeds 10000 to 10199, same scenarios across rows. Standard error on a rate near 80% at n = 200 is about 2.8 points, so differences under ~5 points between two rows are not reliable on their own.

(n/r = recorded in the json, not yet analysed.) Outcome breakdown: nominal 99 success / 1 unsafe site; no-TRN 93 / 6 unsafe / 1 hard landing; low-relief 98 success / 2 hard landings; mixed-hard 188 success / 4 unsafe / 6 hard landings / 2 timeouts; PPO-cov 190 / 2 unsafe / 8 hard; PPO-nocov 197 / 1 unsafe / 1 hard / 1 timeout.

**Reading these.**
- With TRN the filter is consistent (median NEES 6.7 vs. expected ~8.3 for chi-square with 9 dof) and guidance lands ~1 m from target. This is the saturated regime and not interesting for RL.
- Without TRN, position error grows and **the filter is badly overconfident** (median NEES 69; 71% of episodes above 2x dof). Safe-landing rate still looks fine (93%) only because the synthetic terrain is mostly safe (E006).
- On low-relief terrain TRN accepts only ~21% of attempts. The median episode is still fine (4 m) but the tail is bad (p90 204 m, 26% miss the 50 m criterion). The tail is where uncertainty-awareness should matter.

---

## 4. Dated entries

### E001, 2026-10-13: Scope and design choices
- Chose lunar (not Mars) first: no atmosphere, simple dynamics, best DEM coverage.
- Chose 3-DoF + lidar patch matching (not camera/hillshade) first. Camera-based TRN is where sun-angle and shadows break matching; deferred until the pipeline works.
- Decision: the policy observes *estimated* state and *filter covariance*. Truth is used only for dynamics, sensor simulation, reward and evaluation. This keeps the comparison to classical guidance fair.
- Decision: onboard map is a degraded copy of truth (decimate x2) so map error exists even on a perfect-DEM site. Real-DEM runs should instead use a coarser product as the map and a finer one as truth if available.

### E002, 2026-10-13: Environment setup and constraints
- Python 3.13.16, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, gymnasium 1.4.0, pytest 9.1.1, torch 2.14.1, stable-baselines3 2.9.0.
- The PyTorch CPU wheel index was blocked by the sandbox network policy (403); torch installed from the default PyPI instead (large CUDA wheel, runs on CPU). No GPU. 2 CPU cores, so training budgets are modest.

### E003, 2026-10-13: Real planetary DEM is not yet in the loop (**limitation**)
- The sandbox cannot reach NASA PDS Geosciences (`pds-geosciences.wustl.edu`): connection refused at the proxy (403). Real LOLA/SLDEM tiles therefore cannot be downloaded here.
- `terrain.load_dem()` already supports `.npy` (with `dx`) and GeoTIFF/IMG via rasterio, and E016 ran two real PGDA tiles through the `.npy` path. rasterio is not installed.
- **Consequence for any paper:** all numbers in this log are on synthetic terrain. They validate the pipeline and the qualitative trends, not performance on lunar terrain. Real-DEM re-runs are required before claims about the Moon.
- Action needed from user: provide a LOLA/SLDEM2015 tile (e.g. a 5 x 5 km crop near the south pole as GeoTIFF or `.npy` plus cell size).

### E004, 2026-10-13: Synthetic terrain calibration
- Fractal (Hurst 0.85, 18 m RMS relief) plus power-law craters gave median slope 3 to 5 deg and 94 to 98% of cells under 10 deg slope (4 seeds).
- Tried rougher settings (Hurst 0.5 to 0.7, relief 40 to 60 m): median slope 12 to 27 deg and **zero** safe landing targets. Rejected.
- Open issue: the working terrain is *more benign than real lunar polar terrain*, where slopes and boulder fields are a first-order hazard. See E006.

### E005, 2026-10-14: Bugs found and fixed while bringing up the baseline (record for the methods section)
1. **Polynomial guidance hovered instead of landing.** Aiming 3 m above the target with the time-to-go clamp made net acceleration ~0 near zero speed, so the vehicle hung above the target until timeout (11 of 12 smoke-test episodes timed out; 8 of 12 after a first partial fix of the aim point). Pure polynomial guidance stalls at low speed because time-to-go shrinks with altitude. Fix: two-phase guidance with a constant-rate terminal vertical descent. This is standard practice; noting it so the baseline is not accused of being a strawman.
2. **TRN search window cap below 3-sigma (precautionary fix, effect not isolated).** Window was capped at +/-10 map cells (+/-200 m) while 3-sigma of the initial error is 360 m. In one smoke-test episode (seed 9) the true position was ~300 m from the estimate and TRN never produced a fix; I first attributed this to the cap, raised it to 20 cells, and the episode *still* failed, so the cap was not the (only) cause. The cap is now consistent with 3-sigma, but I did not measure its separate effect.
3. **TRN ambiguity test rejected valid matches (this was the cause of the seed-9 failure).** The test excluded only a 2-cell neighbourhood around the correlation peak, but the fractal terrain has a long correlation length, so the main lobe's shoulder looked like a rival peak and nearly every attempt was rejected as "ambiguous". Replaced with a lobe-based test (connected region within `lobe_drop` of the peak, dilated 1 cell; ambiguous if a peak outside it is within `amb_margin`). Sweep of 6 threshold pairs on 150 first-fix attempts at episode start (`scripts/sweep_trn_ambiguity.py`):
   - (0.10, 0.03), the chosen default: 79% accepted, 0% of accepted fixes wrong by more than 60 m, median fix error 5.0 m.
   - Stricter margin (0.10 to 0.10) cut acceptance to 45% with no gain in accuracy, so strict ambiguity rejection was costly here.
   - 20% of first attempts fail the NCC >= 0.7 test under every setting.
   Caveat: thresholds were tuned on the same synthetic terrain used for evaluation. Re-tune on held-out terrain before reporting.

### E006, 2026-10-14: The baseline saturates; "safe landing" is the wrong headline metric here
- Nominal: 99% safe, 97% precise. A learned policy cannot show a large improvement in the nominal regime.
- Because 93 to 98% of synthetic terrain is safe, a lander that is 300 m off target usually still lands safely. Safe-landing rate hides navigation failure (no-TRN: 93% safe but 42% precise).
- Decision: report **precise landing** (safe and within 50 m) as the primary metric and landing-error percentiles as secondary. Add a hazard-dense terrain variant before claiming anything about hazard avoidance.
- Decision: evaluate RL in the degraded regimes (no TRN after some altitude, low relief, larger initial error), where the baseline has headroom.

### E007, 2026-10-14: Filter consistency finding
- With TRN off, NEES median is 69 against an expected ~8.3 (dof 9). The EKF reports small sigma while real error grows. **Hypothesis (not yet tested):** the terrain-aided altimeter update linearises a nonlinear terrain model, so with a large horizontal error it shrinks horizontal sigma along map gradients without real absolute information. Seen directly in one TRN-denied smoke episode: sigma fell from 120 m to ~19 m while true horizontal error grew to ~410 m. Test: disable the altimeter's terrain aiding (use a flat-ground height model) and re-measure NEES.
- Worth a short section in a paper: a policy that conditions on covariance is only as good as the covariance. Planned follow-up: inflate the altimeter measurement noise as a function of local map curvature and position uncertainty, and report NEES again. Until then, H1 results in no-TRN scenarios are confounded by an overconfident filter.

### E008, 2026-10-14: Unit tests (17 passing) and one test-design lesson
Tests cover: slope of a known plane, bilinear interpolation and axis order, DEM downsample registration, terrain determinism, hazard map, free-fall and rocket-equation checks, thrust/tilt limits, that the observation is built from the estimate (not truth), covariance ablation, EKF covariance symmetry and PSD at every step, NEES consistency with TRN, TRN beating dead-reckoning (>=4x lower error), residual-wrapper equivalence to baseline, reset reproducibility.
- Lesson: my first flat-terrain test expected a "flat patch" rejection, but at 600 m altitude lidar noise (1.5 m) exceeds the 0.3 m flat-patch relief threshold, so that gate only fires near the ground. TRN still refused the fix (correlation threshold), which is the behaviour that matters. The test now asserts the real requirement: 0 accepted fixes in 60 attempts on 0.1 m-relief terrain. Design note for the paper: relief gating should scale with sensor noise; currently the NCC threshold does the work at altitude.

### E009, 2026-10-14: Benchmark definition (`trnrl/benchmarks.py`)
- Scenarios: `nominal`, `no_trn`, `low_relief`, `big_init` (300 m initial error), `mixed_hard`.
- **Train/test split by terrain**: training uses site seeds 2000+ (16 sites), evaluation uses 1000+ (8 sites, never trained on). The first four baseline rows in the registry used the test sites; the baseline has no learned parameters so this is only a bookkeeping point.
- `mixed_hard` is the target regime: baseline precise-landing 75.5% with a heavy tail (p90 168 m).
- Caveat: ambiguity thresholds (E005) were tuned on the *test* sites' neighbours of the same generator family. When writing the paper, re-tune on training sites only.

### E010, 2026-10-14: RL design decision, residual policy on top of the baseline
- Pure PPO from scratch: 40k steps (135 s, ~300 steps/s on one core) gave 0% safe landings. Landing from a random policy is a hard exploration problem and typically needs millions of steps; at this speed that is hours per run on 2 cores.
- Chosen: **residual RL**. Action = clip(baseline action + 0.5 x policy action). The baseline action is appended to the observation so the policy can learn *when* to deviate. Policy log-std init -1.5 so training starts near baseline behaviour. Verified by test that a zero residual reproduces the baseline exactly.
- Trade-off to state in the paper: this is not end-to-end RL; improvement is measured over a strong classical prior. It is closer to how a learned component would actually be fielded in GNC.
- Reward: potential shaping toward the **true** target (privileged information, standard asymmetric training) + fuel/time costs + terminal bonus/penalty on true landing error. The policy's *observation* uses only the estimate.
- PPO: 8 envs, 512 steps x 8, batch 512, 8 epochs, lr 3e-4, gamma 0.995, GAE 0.95, MLP 2x256 (separate actor/critic), reward scaled x0.05.
- Runs launched: `cov` and `nocov` (identical except the covariance/fix-age features are zeroed in `nocov`), seed 0, 600k steps, `mixed_hard`, train split. **One seed is not enough to claim H1**; more seeds needed.

### E011, 2026-10-14: Altimeter coupling test: hypothesis from E007 was *half* right
Added `NavConfig.alt_mode`: `coupled` (default; EKF with map-gradient coupling, used in all runs above and in the PPO runs of E010) vs `decoupled` (height measurement only updates the vertical state; height error caused by horizontal uncertainty, `gx^2 Pxx + gy^2 Pyy`, is added to the measurement noise, Schmidt-style).

| Baseline guidance, 100 episodes (seeds 10000+) | Precise (<=50 m) | Median / p90 landing err | Median NEES | Episodes with median NEES > 18 |
|---|---|---|---|---|
| `no_trn`, coupled | 42% | 69 m / 336 m | 69.4 | 71% |
| `no_trn`, decoupled | **9%** | 153 m / 235 m | 15.3 | 38% |
| `mixed_hard`, coupled (first 100 of the 200-episode run) | 77.0% | n/r | 8.1* | 18.5%* |
| `mixed_hard`, decoupled | **84.0%** | 1.3 m / 135 m | 7.2 | 1% |

(*from the 200-episode run.)

What this shows:
- **Without TRN, the coupled altimeter carries real positioning information** (terrain-contour matching along map gradients): turning coupling off drops precision 42% to 9%. So the coupled filter's overconfidence (NEES 69) is not just a linearisation artifact; part of what it extracts is genuine. The decoupled filter is much better calibrated (NEES 15) but loses that information.
- **With intermittent TRN (`mixed_hard`), decoupling helps** (+7 points precise, p90 168 to 135 m). Mechanism consistent with the data: over the same 100 seeds, TRN innovation-gate rejections fell from 72 to 9 and the fix acceptance rate rose from 0.58 to 0.61. An overconfident filter rejects good fixes because its own sigma is too small. This is a mechanism supported by the counts, not a controlled causal test.
- Caveats: n = 100 per cell; no significance test yet; the 7-point gap is on paired seeds but within plausible noise at this n (a 200-episode decoupled run is queued for when training frees the CPU).
- **Consequence for the RL comparison:** the strongest classical stack is *baseline guidance + decoupled altimeter*. A learned policy must be compared against that, not only against the original baseline, or any RL gain may just be recovering the filter fix. The two PPO runs in E010 were launched with the coupled filter (their module state predates this change); they remain a valid comparison *within* the coupled-filter environment.
- A natural third option (not built): keep coupling but inflate its measurement noise by a curvature/uncertainty-dependent term, to keep the extra information while restoring calibration.

### E012, 2026-10-14: Where the misses come from (diagnostic, all on `mixed_hard` test)
Precise-landing rate split by whether TRN produced at least one accepted fix:

| Policy | Zero-fix episodes: precise (n) | At least one fix: precise (n) |
|---|---|---|
| baseline (coupled) | 7% (42) | 94% (158) |
| baseline (decoupled) | 6% (32) | 98% (168) |
| PPO with covariance | 10% (52) | 91% (148) |
| PPO without covariance | 17% (47) | 97% (153) |

- For the baseline, |landing error - final navigation error| has median 0.1 to 0.2 m (p90 0.3 to 1.0 m). Given its estimate the guidance lands essentially exactly where it believes the target is, so **every miss is a navigation miss**, not a guidance miss.
- By site and blackout (baseline, decoupled filter, n = 200): rough sites 0% zero-fix and 97 to 98% precise *with or without a blackout*; low-relief sites with blackout 46% zero-fix and 56% precise; low-relief without blackout 18% zero-fix and 80% precise. **All of the headroom is on low-relief terrain.**
- TRN is only usable above 150 m AGL; the baseline drops below that at a median of 70 s (20 episodes), so fix opportunities exist only in roughly the first 70 s. A blackout starting early can remove all of them.
- Implication: terminal-guidance learning cannot help. A learned (or rule-based) component can only help by changing *whether a fix is obtained*: wait out a blackout (costs fuel and time) or detour toward textured terrain (not yet tested).

### E013, 2026-10-14: Covariance-triggered hold (classical, hand-written)
`WaitGuidance`: while nav horizontal sigma > 40 m, AGL > 500 m, cumulative hold < `max_wait`, and propellant above a 220 kg reserve, hover instead of descending; resume when a fix reduces sigma. Uses only filter outputs.
- First attempt failed (113 of 200 timeouts): after hovering, the polynomial guidance computes time-to-go from the (near zero) descent speed and plans an extremely slow descent. Fix: assume a nominal 15 m/s descent speed for time-to-go when above 300 m AGL (hold variant only; baseline untouched). Record this: it is the same stall mechanism as E005.1.
- `max_wait` chosen on the **training** split (200 episodes, decoupled filter), then evaluated once on test:

| max_wait [s] | 0 | 30 | 50 | 70 | 80 | 90 |
|---|---|---|---|---|---|---|
| Precise (train) | 82.5% | 86.5% | 87.5% | **90.5%** | 89.0% | 81.0% |
| Timeouts (of 200) | 0 | 0 | 0 | 0 | 5 | 20 |
| p90 landing err [m] | 135 | 89 | 85 | 18 | 11 | 31 |
| Median fuel [kg] | 141 | 179 | 199 | 218 | 225 | 172 |

  The cliff at 80 s is the 250 s mission-time cap (hold + descent no longer fits). 70 s was selected.
- **Test result (200 episodes, never used for tuning):** decoupled filter 83.0% -> **89.0%** precise, p90 110 -> 60 m, zero-fix episodes 32 -> 21, fuel 152 -> 220 kg (+68 kg of ~500 kg usable). Coupled filter 75.5% -> **80.5%**, but p90 stays high (246 m).
- Disclosure: before the train-split sweep I ran one 90 s hold on the **test** split (`results/wait_mixed_hard_dec.json`: 79.0% precise, 26 timeouts). That run is what prompted sweeping shorter holds, so the test split was not perfectly untouched, although the value (70 s) was selected on the training split and the test number above was computed once.
- Caveats: the hold rule is hand-designed and its single parameter tuned on training sites; the 250 s cap is an arbitrary mission-time constraint that directly shapes the optimum; fuel cost is real and not part of the "precise landing" metric.

### E014, 2026-10-14: First PPO results: **negative**
Training: 600k steps each, ~28 min per run, seed 0, `mixed_hard` train split, coupled filter. Training-set precise-landing (rolling 500 episodes, stochastic policy) was ~0.77 at 30k steps and 0.79 (cov) / 0.81 (nocov) at 600k: essentially flat. I did not measure the baseline on the coupled *train* split, so I cannot say whether that small rise is learning or noise.

Held-out test, 200 episodes, compared with the baseline in the *same* (coupled) environment:

| | Precise | p90 err | Fuel | Hard landings |
|---|---|---|---|---|
| baseline | 75.5% | 168 m | 152 kg | 6 |
| PPO, no covariance | 78.0% | 218 m | 147 kg | 1 |
| PPO, with covariance | 70.0% | 230 m | 119 kg | 8 |
| covariance-triggered hold (E013) | 80.5% | 246 m | 220 kg | 2 |

- **H1 (covariance in obs helps): not supported.** `cov` is 8 points *below* `nocov`. One seed each and SE of the difference is ~4.4 points, so this is not decisive either way; it certainly provides no support.
- **H2 (RL beats fixed guidance on p90): not supported.** p90 is worse for both PPO runs than for the baseline (230 and 218 vs 168 m).
- The `cov` policy used less fuel (119 kg) and had more hard landings (8): it learned to trade safety margin for propellant, which the reward allows.
- The hand-written hold rule beats both learned policies on precise landing, and the decoupled-filter variant of it beats everything (89.0%).
- **Why PPO did not find the hold behaviour is untested.** Candidate explanations, none verified: (a) credit assignment over a 60 to 100 s delayed benefit with a sparse terminal signal; (b) residual authority k = 0.5 and small initial policy noise confine it near the baseline; (c) 600k steps is a small budget; (d) the coupled filter's overconfident covariance is a poor input feature (E007, E011). The last is the cheapest to test (retrain on the decoupled filter).
- **Do not generalise from this.** One seed, one reward design, one budget. This is a "first attempt did not work", not "RL cannot help here".

---

### E015, 2026-10-14: Figure set for non-technical audiences (no new experiments)
Built `scripts/make_figures.py` (`python scripts/make_figures.py [terrain navigability how descent scorecard miss why waiting]`), which regenerates `figures/01` to `08` from `results/*.json` and the simulator. Each figure prints the numbers it plots (`data:` lines); I checked them against the results files. A plain-language page summarises them.
- Figures: 3D terrain; where terrain matching works; how it works; one descent step by step; scorecard; miss distance; why landings miss; waiting trade-off.
- **Caveats carried into the figures:** all terrain is synthetic; fig 04 uses hand-picked example episodes (seeds 10001, 10051 on the test split) for illustration, not a sample; fig 08 is chosen on the train split and reports the test result in its footer; PPO is one seed per variant and negative (E014).
- No new claims are made here. Status of "3D visualisation" in the checklist: partly done (static 3D terrain only; no uncertainty-map view yet).

### E016, 2026-10-14: First runs on real lunar terrain (NASA PGDA LOLA 5 m DEMs)
**Data.** Two 8 km x 8 km tiles cut from the PGDA high-resolution LOLA polar DEMs (Barker et al.; `Site04_final_adj_5mpp_surf.tif` Shackleton rim, `Site23_final_adj_5mpp_surf.tif` Malapert massif), block-averaged from 5 m to 10 m cells with `scripts/crop_dem.py` and stored as `data/site04.npy`, `data/site23.npy`. Site 04 = train split, Site 23 = test split (`build("real+dec", split)`, blackout probability 0.4 as in `mixed_hard`, decoupled altimeter).
**Measured properties of the tiles.** Height std 314 m (Site 04) and 271 m (Site 23) against about 18 m RMS for the synthetic tiles; roughly 14 to 35% of cells have slope under 10 deg; the 20 m onboard map differs from the 10 m truth by 0.33 m and 0.28 m RMS. Candidate landing targets: 164 (Site 04) and 150 (Site 23), clustered in a small part of each tile. The data page states that about 90% of the pixels are interpolated between laser tracks, so these surfaces are smoother at small scales than real ground.
**Results (100 episodes each, one tile per split; SE about 2 points).**

| Policy, tile | Precise (<=50 m) | Safe | Median error | Zero-fix episodes | Median fuel |
|---|---|---|---|---|---|
| baseline, Site 23 (test) | 95% | 96% | 1.2 m | 0 | 149 kg |
| hold rule (70 s), Site 23 (test) | 95% | 96% | 1.3 m | 0 | 142 kg |
| baseline, Site 04 (train) | 98% | 98% | 1.2 m | 0 | 130 kg |

- **Measured:** on these two tiles terrain matching produced a fix in every episode, even with blackouts, and the baseline already lands 95 to 98% precisely. The hold rule changes nothing here (95% vs 95%).
- **Hypothesis, not tested:** the tiles have kilometre-scale relief 15 times larger than the synthetic sites, so terrain matching is rarely ambiguous. This is consistent with E012 (all headroom is on low-relief ground) but does not prove it; I did not vary the tile while holding everything else fixed.
- **What this does not show:** (a) one tile per split, with targets clustered in one area, so the result describes two places, not lunar terrain in general; (b) the interpolated surfaces may make matching easier than on real ground; (c) these are steep crater-rim and massif sites, the opposite of the flat low-relief case that the hold rule was built for. A flat crater floor (for example Haworth) is the informative next test.
- PPO was not run on real tiles.

### E017, 2026-10-14: Landing sandbox and random-window RL design (code only, no results yet)
Built tooling, no new experiment. **Nothing here is evidence about RL performance yet.**
- `trnrl/dem_source.py` (`BigDEM`): lazy GeoTIFF/.npy reader, overview, 8 km windows block-averaged to 10 m cells; geographic split: left 70% of the map by columns = train, right 30% = held-out test; train and test windows never overlap. `--frac 1.0` disables the split inside one map so training and testing can use different site files (needed for small PGDA tiles; also a stronger held-out test).
- `trnrl/windows.py`, `scripts/train_ppo_windows.py`, `scripts/eval_windows.py`: PPO residual policy trained on a pool of random train-region windows (default 48), evaluated on random held-out-region windows with the same episode seeds for every policy. Settings match the strongest classical stack so far: decoupled altimeter, 40% blackout probability.
- `sandbox/` (`server.py`, `index.html`, vendored three.js): local app to pick any window on the map and fly baseline, hover-and-wait and RL together in 3D; overlays for safe ground and where terrain matching works; ring = navigation uncertainty.
- Tests: `tests/test_windows.py` (window orientation, no train/test overlap, targets exist); 20 tests pass.
- **Smoke test only:** a 24k-step PPO run on a synthetic 32 km map trained and evaluated end to end (12 episodes, 4 windows); numbers from it are meaningless and are not reported.
- **Open design points:** (a) the split is by columns, so a held-out strip next to the train area shares terrain character; a stronger test would hold out a different geologic unit or a different site. (b) PGDA tiles are about 90% interpolated, so windows may be smoother than real ground (E016). (c) The earlier negative PPO result (E014) used synthetic terrain; training on varied real windows addresses data diversity, not the credit-assignment problem for hover decisions, which remains the hold/detour experiment.
- **Next:** train 3 seeds x (cov, nocov) on random windows, evaluate paired on held-out windows, compare with baseline and the hold rule.

### E018, 2026-10-14: PPO retrained on the decoupled filter (test of E014 explanation d): still negative
Same setup as E014 (residual PPO, k = 0.5, 600k steps, 8 envs, seed 0, `mixed_hard` train split) except the navigation filter is the decoupled altimeter (`mixed_hard+dec`). Held-out test split, 200 episodes, deterministic policy, compared in the same environment:

| | Precise | Safe | p90 err | Fuel | Zero-fix episodes |
|---|---|---|---|---|---|
| baseline | 83.0% | 99.0% | 110 m | 152 kg | 32 |
| PPO, no covariance | 82.5% | 98.5% | 104 m | 132 kg | 34 |
| PPO, with covariance | 59.0% | 74.5% | 161 m | 114 kg | 41 |
| hold rule 70 s (E013) | 89.0% | 99.5% | 60 m | 220 kg | 21 |

- **Measured:** PPO without covariance matches the baseline on precise landing (82.5 vs 83.0%, SE about 2.7 points each, so no difference) with about 20 kg less fuel. PPO with covariance is much worse: 50 of 200 episodes ended in a hard landing.
- **E014 explanation (d) "overconfident coupled covariance" is not supported:** with a calibrated-covariance filter the covariance variant got worse, not better. The conclusion is limited to one seed per variant.
- **Training curves** (rolling 500 stochastic episodes) ended at 0.72 precise (cov, falling in the last 100k steps) and 0.88 (nocov), so the covariance run did not converge stably; the gap between the nocov training figure and its deterministic test figure was not investigated.
- **Hypothesis, not tested:** the reward lets the policy trade touchdown safety for propellant (the cov policies used least fuel and had most hard landings in both E014 and E018). It could also be that covariance features let the policy overfit to training terrain. Neither was checked.
- **Where this leaves RL:** two experiments (E014, E018), one seed each, both fail to beat the hand-written hold rule, and neither learned to hover. Residual control around the baseline does not find the hold behaviour. The next RL experiment should change what the policy decides (a high-level hold/detour choice), not train longer.

### E019, 2026-10-15: Random-window PPO on real PGDA terrain, training curve only (run on the user's machine)
User ran `train_ppo_windows.py <Site TIFF> cov 0 600000 48` (48 random 8 km training windows, 600k steps, 574 s). I have the printed training log only; **no held-out evaluation exists yet** and I have not seen the TIFF or the checkpoints.
- **Measured (rolling 500 episodes, stochastic policy, training windows):** precise landing 0.92 at 30k, 0.97 at 90k, about 0.94 through 300k, falling to 0.81 by 420k to 450k, a partial recovery to 0.90 at 510k, then 0.78 at 570k and **0.65 at 600k** (safe landing equals precise, 0.65). The saved final model is therefore the weakest point of the run.
- **Context:** the baseline alone landed 95 to 98% precisely on the two real tiles (E016), so the first 300k steps are consistent with a policy close to the baseline and little headroom. The later fall below that level means the learning updates degraded it. One seed, `cov` variant only.
- **Hypotheses, not tested:** (a) late PPO instability (policy-std or value-function drift, large updates on a nearly saturated reward); (b) the covariance inputs overfit the training windows (the synthetic runs of E014 and E018 also showed `cov` worse than `nocov`); (c) the drop is an artefact of the logging window. None checked.
- **Tooling response:** `train_ppo_windows.py` now saves a checkpoint at every log step and a best-by-training-score checkpoint (`CkptLog`); the best-by-training choice is optimistic (stochastic policy, training windows).
- **Next:** evaluate `_best`, an early checkpoint and the final model on held-out windows with `eval_windows.py` against baseline and hold rule before drawing any conclusion.

### E020, 2026-10-15: Random-window PPO on a held-out real site: no gain over the baseline; late training degrades
Run on the user's machine (I have the printed summaries, not the JSON files). Train: 48 random 8 km windows of the Site 04 PGDA DEM (`--frac 1.0`), 600k steps, seed 0, `cov` and `nocov` variants, checkpoints every 30k steps. Test: 200 episodes over 24 random windows of the **Site 23** DEM (a different site, never used in training), decoupled filter, 40% blackout probability, same windows and seeds for every policy. SE of a precise-landing rate near 97% at n = 200 is about 1.1 points.

| Policy | Precise | Safe | Hard / unsafe / timeout | p90 err | Fuel | Zero-fix |
|---|---|---|---|---|---|---|
| baseline | 97.5% | 98.0% | 1 / 3 / 0 | 2.3 m | 148 kg | 1 |
| hold rule 70 s | 97.5% | 98.0% | 1 / 3 / 0 | 2.6 m | 141 kg | 1 |
| PPO cov, step 60k | 97.0% | 97.5% | 1 / 3 / 1 | 6.0 m | 156 kg | 1 |
| PPO cov, step 300k | 97.0% | 98.0% | 1 / 3 / 0 | 11.3 m | 140 kg | 2 |
| PPO cov, best-by-training | 97.5% | 98.5% | 1 / 2 / 0 | 11.9 m | 137 kg | 2 |
| PPO cov, final (600k) | 85.5% | 86.5% | 25 / 2 / 0 | 16.1 m | 120 kg | 2 |
| PPO nocov, step 60k | 95.0% | 95.5% | 1 / 2 / 6 | 4.5 m | 153 kg | 1 |
| PPO nocov, step 300k | 98.0% | 98.5% | 1 / 2 / 0 | 7.6 m | 148 kg | 1 |
| PPO nocov, best-by-training | 97.0% | 98.0% | 0 / 4 / 0 | 7.2 m | 141 kg | 2 |
| PPO nocov, final (600k) | 95.0% | 96.0% | 5 / 2 / 1 | 13.2 m | 128 kg | 2 |

Training-curve endpoints (rolling 500 stochastic training episodes): cov 0.71 precise, nocov 0.91.
- **Measured:** no PPO checkpoint beats the baseline; all differences among baseline, hold rule and the good checkpoints are within one standard error. On this site the baseline already lands within 2.3 m (p90) of the target when it lands at all, so navigation is not the limiting factor.
- **Measured:** the final cov model is clearly worse (85.5%, 25 hard landings); the final nocov model is mildly worse (95.0%, 5 hard landings). Late training degrades the policy in this setup. Evaluating only the final checkpoint would have misreported RL.
- **Measured:** across cov checkpoints, median fuel falls (156, 140, 137, 120 kg) as hard landings rise at the final one; PPO's p90 landing error is 2 to 7 times the baseline's. Consistent with the earlier hypothesis that the reward lets the policy trade touchdown safety for propellant (E014, E018), not proven; one seed.
- **Where the remaining headroom is:** the baseline's 2.5 points of loss are 3 `unsafe_site` landings and 1 hard landing, not navigation misses. Whether those are avoidable with the onboard (20 m) map is untested; the hazards may be invisible at that resolution.
- **Limits:** one seed per variant; test is one site (Site 23) and PGDA surfaces are about 90% interpolated, likely smoother than real ground; windows within one DEM overlap, so the 24 windows are not independent samples of terrain; the best-by-training checkpoint was picked on training windows only. The hold rule also gives no benefit here because TRN almost always fixes (zero-fix episodes 1 of 200).
- **Conclusion for the project:** on rugged real terrain there is little for any policy to improve in navigation, and the residual PPO as built adds nothing. An RL result needs a setting with real headroom: flat low-relief real terrain (navigation misses, E012), or the hazard-avoidance lever (choosing a safer touchdown point within the 50 m precision zone).

### E021, 2026-10-15: Haworth (PGDA, 29.8 x 29.8 km) with the proper geographic split: no headroom there either
Run on the user's machine; I unpacked the JSON/CSV bundle (`results/haw_*.json`, `results/summary_windows.csv`) and checked the table below against it. Haworth map: relief std 704 m over the whole map (crater walls included), about 50% of the map under 10 deg slope in a coarse 33 m view (this understates fine-scale roughness). Split: left 70% of columns = train, right 30% (8.9 km wide, fits one 8 km window) = held-out test; 24 random windows, 200 episodes, decoupled filter, 40% blackouts. SE about 0.5 points at 99.5%, 1.4 at 96%, 2.5 at 85%.

| Run | Precise | Hard / unsafe / timeout | p90 err | Fuel |
|---|---|---|---|---|
| baseline, Haworth held-out strip | 99.5% | 0 / 0 / 0 | 2.1 m | 147 kg |
| hold rule 70 s, held-out strip | 99.5% | 0 / 0 / 0 | 2.2 m | 141 kg |
| baseline, Haworth train region | 96.0% | 0 / 7 / 0 | 2.1 m | 150 kg |
| hold rule, train region | 96.5% | 0 / 6 / 1 | 2.1 m | 142 kg |
| Site 04 PPO (zero-shot), cov best / step 300k | 98.5% / 98.5% | 0 / 2 / 0 each | 11.5 / 10.5 m | 143 / 142 kg |
| Site 04 PPO (zero-shot), nocov best / step 300k | 96.5% / 99.0% | 0 / 1 / 5 ; 0 / 0 / 1 | 6.0 / 6.6 m | 151 / 153 kg |
| Haworth PPO cov, best / step 300k / final, on held-out strip | 99.5% / 85.0% / 89.5% | 0/0/0 ; 28/0/0 ; 0/0/19 | 3.7 / 9.3 / 13.8 m | 143 / 119 / 147 kg |
| Haworth PPO nocov, best / step 300k / final, on held-out strip | 99.5% / 93.0% / 94.0% | 0/0/0 ; 0/1/13 ; 8/2/0 | 3.4 / 7.2 / 9.3 m | 155 / 156 / 118 kg |
| Haworth PPO cov, best / step 300k / final, on Site 23 | 97.5% / 91.5% / 82.5% | 1/3/0 ; 12/3/0 ; 1/3/30 | 3.7 / 9.8 / 12.7 m | 144 / 126 / 158 kg |
| Haworth PPO nocov, best / step 300k / final, on Site 23 | 97.5% / 94.5% / 89.5% | 1/3/0 ; 1/4/4 ; 16/3/0 | 3.5 / 6.6 / 9.3 m | 153 / 151 / 116 kg |

- **Measured:** on the held-out Haworth strip the baseline is at 99.5% with 1 zero-fix episode of 200, and the hold rule changes nothing. No policy beats the baseline in any row; the best-by-training checkpoints tie it (99.5%, 97.5%), later checkpoints are worse, and final checkpoints are the worst, as in E019, E020. Failure modes differ by checkpoint: hard landings (cov step 300k, 28 of 200) or timeouts (cov final, 19; nocov step 300k, 13).
- **Measured:** three real sites now (Site 04, Site 23, Haworth): baseline precise landing 95 to 99.5%, TRN almost always fixes. The low-relief, no-fix regime that gives 83% on the synthetic `mixed_hard` benchmark does not occur in these real tiles.
- **Hypothesis, not tested:** real polar DEMs have hundreds of metres of large-scale relief, so terrain matching is rarely ambiguous at the 20 m onboard-map scale; the DEMs are about 90% interpolated, which may also affect matching.
- **Limits:** one seed per variant; Haworth held-out strip and train region are parts of one map (possible terrain-type shift, not inspected); windows overlap within a map.

### E022, 2026-10-15: Where is the headroom on real tiles? Stressing navigation (run here, `scripts/stress_real.py`)
100 episodes per cell on the two 8 km tiles (one window each; `site04` = train split, `site23` = test split), decoupled filter. Settings: default (40% blackouts of 30 to 70 s); big_init (300 m initial position error); long blackout (every episode, 60 to 100 s); both. SE up to about 4 points.

| Tile | Setting | Baseline precise (safe) | Hold-rule precise (safe) | Baseline outcomes (non-success) |
|---|---|---|---|---|
| site04 | default | 98% (98) | 98% (98) | 1 unsafe, 1 hard |
| site04 | big_init | 83% (90) | 83% (88) | 8 timeout, 2 unsafe |
| site04 | long blackout | 99% (99) | 99% (99) | 1 unsafe |
| site04 | both | 82% (89) | 82% (88) | 8 timeout, 3 unsafe |
| site23 | default | 95% (96) | 95% (96) | 4 unsafe |
| site23 | big_init | 78% (80) | 78% (80) | 3 timeout, 13 unsafe, 4 hard |
| site23 | long blackout | 94% (96) | 94% (96) | 4 unsafe |
| site23 | both | 77% (81) | 78% (82) | 5 timeout, 10 unsafe, 4 hard |

- **Measured:** long blackouts do nothing on these tiles (terrain matching still fixes; zero-fix episodes 0 to 1 of 100), so the hold rule has nothing to do. A 300 m initial position error cuts precise landing by about 15 to 18 points on both tiles.
- **Measured:** under big_init the baseline's landing error when it does land stays small (p90 3 to 5 m) and zero-fix episodes are 0 to 1; the losses are timeouts, unsafe touchdowns and hard landings. **The failures are guidance failures after a late 300 m position correction, not navigation failures.**
- **Hypothesis, not tested:** after the first terrain fix jumps the estimate by up to 300 m, the polynomial guidance has too little time, tilt authority or altitude to re-target gracefully, and ends up timing out or touching down off-nominal. I did not inspect trajectories.
- **Implication:** this is a regime with 15 to 22 points of headroom on real terrain where RL has a natural role (guidance under a large late correction). The residual PPO has not been tried here. Earlier residual-PPO failures (E014, E018, E020) were in regimes where the baseline was at 83% (navigation-limited) or 95%+ (no headroom).
- **Limits:** one window per tile; 100 episodes per cell; the two tiles are steep massif / rim terrain; big_init is a synthetic stress, not a measured navigation error.

### E023, 2026-10-15: CORRECTION to E022. The big_init failures were a search-window limit, not guidance
Hypothesis from the code comment (`search_max_cells` "must cover 3*init_pos_h / map dx"): the default window is 20 cells = 400 m at the 20 m onboard map, which does not cover 3 sigma once init_pos_h >= 133 m. Test: `scripts/sweep_init.py`, baseline guidance, 100 episodes per cell, sigma of the initial horizontal position error 120 / 300 / 600 / 1000 m, search window either 20 cells (old) or widened to ceil(3*sigma/20)+2 cells. Start range unchanged (1.0 to 1.6 km from target). SE up to 4 points.

| Tile | sigma | Precise, narrow (20 cells) | Precise, wide (cells) | Wide: non-success outcomes, zero-fix episodes |
|---|---|---|---|---|
| site04 | 120 | 98% | 98% (20) | 1 unsafe, 1 hard; 0 |
| site04 | 300 | 83% | 99% (47) | 1 unsafe; 0 |
| site04 | 600 | 24% | 99% (92) | 1 unsafe; 0 |
| site04 | 1000 | 12% | 99% (152) | 1 unsafe; 0 |
| site23 | 120 | 95% | 95% (20) | 4 unsafe; 0 |
| site23 | 300 | 78% | 92% (47) | 4 unsafe, 2 timeout, 1 hard; 0 |
| site23 | 600 | 23% | 73% (92) | 15 unsafe, 3 timeout, 3 hard; 17 |
| site23 | 1000 | 11% | 80% (152) | 8 hard, 6 unsafe, 4 timeout; 9 |

- **Measured:** the narrow window explains the E022 big_init losses on site04 (83% to 99% when widened) and most of them on site23 (78% to 92%). With the narrow window the landing error is hundreds of metres at sigma >= 600 (p90 about 450 to 580 m): the matcher cannot find the true position.
- **Measured:** with a properly sized window, site04 is at 99% up to 1000 m initial error. Site23 degrades at 600 to 1000 m (73 to 80%) with 9 to 17 zero-fix episodes of 100 and unsafe or hard touchdowns.
- **Retraction:** the E022 statements that the big_init failures were "guidance failures after a late correction" and "the first real-terrain regime with RL headroom" are withdrawn for site04 and weakened for site23. That interpretation was my hypothesis, and the E023 data contradicts it. The E022 table itself (narrow window, 300 m) is correct as a measurement of that configuration.
- **Hypothesis, not tested:** on site23 at large initial error, a bigger search area makes ambiguous matches more likely (zero-fix and unsafe cases rise), so the remaining loss could be a navigation-ambiguity problem where a hold or detour decision might help. Trajectories not inspected.
- **Limits:** one window per tile; 100 episodes per cell; sigma values of 600 to 1000 m with a 1.0 to 1.6 km start are an extreme setting that I did not check against real mission numbers (the real pre-TRN error was not found in the sources I read); runtime grows with search size (a wide cell takes minutes).

### E024, 2026-10-15: Phase 1 diagnosis. Why does baseline fail on site23 at large initial error?
`scripts/diagnose_failures.py data/site23.npy <sigma> 60 [tag] [policy]`: 60 episodes (seeds 10000 to 10059), widened TRN search window, every non-precise episode classified from its history (rules in the script docstring). SE about 6 points. Files: `results/diag_site23_{600,1000}[_wait].json`.

| Setting | Precise | zero_fix (ambiguous, no fix) | false_fix (>100 m wrong fix accepted) | nav_ok_hazard (nav error <= 15 m, hard or unsafe touchdown) | other (mostly timeouts) |
|---|---|---|---|---|---|
| baseline, sigma 600 | 43 / 60 (72%) | 10 | 2 | 3 | 2 |
| baseline, sigma 1000 | 51 / 60 (85%) | 1 | 2 | 5 | 1 |
| hold rule 70 s, sigma 600 | 39 / 60 (65%) | 9 | 2 | 6 | 4 |
| hold rule 70 s, sigma 1000 | 42 / 60 (70%) | 2 | 1 | 7 | 7 (+1 nav_bad) |

- **Measured:** two different failure types. (A) Ambiguity, no fix: at sigma 600 the biggest bucket (10 of 17 failures); all 10 were `rejected_ambiguous` throughout, and the lander touches down 470 to 1220 m off target on unsafe ground (9 of 10). (B) Navigation fine, touchdown bad: 3 to 5 episodes at sigma 600 and 1000 (nav error at touchdown <= 15 m, but hard landing or unsafe site), plus 1 to 2 timeouts with good navigation. Type B is a terminal-phase problem, not navigation. (C) False fixes (a wrong match accepted): 1 to 2 per 60 at both settings; these end 900 to 1900 m off.
- **Measured:** the hold rule does not help in either setting: 65% vs 72% at sigma 600 and 70% vs 85% at sigma 1000, with more timeouts and hard landings. It leaves the ambiguity bucket unchanged (9 vs 10).
- **Measured:** the sweep (E023) counted zero-fix episodes over all episodes (9 at sigma 1000); here "zero_fix" counts only non-precise episodes (1 at sigma 1000). The difference is lucky precise landings without any fix, not a conflict.
- **Hypothesis, not tested:** hovering keeps the same viewpoint, so it does not resolve an ambiguous match; a change in altitude or lateral position might. The type-B touchdown failures may come from how the baseline picks and keeps its target near hazards; I did not inspect the target or hazard geometry.
- **Implication for the plan:** the hold/duration idea targets type A, but the hand-written hold already fails there, so a learned version of the same idea has a poor prior. Type B (good navigation, bad touchdown) appears at every initial error setting I looked at, is a pure guidance problem, and is where residual or retargeting RL has a plausible role.
- **Limits:** one tile, 60 episodes per cell, the largest sigma values are untested against real mission numbers, classification thresholds (100 m, 15 m, 300 m) are my choice.

### E025, 2026-10-15: Phase 2 step 1. What exactly happens in the "navigation fine, touchdown bad" episodes?
`scripts/inspect_typeB.py` re-ran the type B and timeout episodes of E024 (site23, baseline, seeds 10035, 10037, 10038, 10055, 10056 at sigma 600; 10004, 10005, 10035, 10037, 10049, 10056 at sigma 1000) and printed touchdown speeds, target safety on the 10 m truth, and fuel. Eleven episodes; small sample, read as case studies, not rates.

| Pattern | Seeds | Evidence |
|---|---|---|
| Designated target is itself unsafe in truth | 10035 (both settings) | landed within 1 m of the target, slope at target 10.1 deg vs 10 deg limit; targets are chosen safe on the 20 m onboard map only (`Site.__init__`), truth is the 10 m map |
| Reaches the target, hits at 4+ m/s | 10037 (both), 10049 | landing error 6 to 9 m, vz 4.0 to 4.5 m/s (limit 2), 360 kg of fuel left |
| Crashes short of the target at speed | 10056 (both), 10004 | t = 55 to 62 s, 150 to 250 m from target, vh 18 to 22 m/s, vz 6 m/s, 423 kg fuel left, first fix at t = 19 s (10056) |
| Never lands, loiters until the time limit | 10038, 10055 | timeout at t = 250 s, 27 to 57 m above ground, vz 0.5, 230 kg fuel left |

- **Measured:** in none of these did the lander run out of fuel (230 to 420 kg left). The failures are not navigation (position error at the end 1 to 15 m in all but 10004, 10005, 10049).
- **Measured:** the baseline enters its terminal descent only if altitude is under 40 m and the estimated horizontal offset is under 25 m (`PolyGuidance.act`); otherwise it keeps flying the polynomial gate law (gate 30 m above target, vz -3 m/s) with a time-to-go that collapses near the ground.
- **Hypothesis, not tested:** the hard landings at the target and the loitering both come from arriving over the gate with a horizontal offset above 25 m (after a late position correction), so terminal descent never starts and the polynomial law either dives at 4+ m/s or hovers; the short crashes come from a late correction leaving too little altitude and tilt authority to close hundreds of metres. I did not plot trajectories.
- **Implication:** type B is largely a deficiency of the hand-written guidance, which a better classical rule (abort to a hover and re-approach when the offset is above the gate, climb-and-redirect after a big correction) could plausibly fix without RL. A fair comparison therefore needs a "baseline v2" with that rule before claiming anything for RL; if baseline v2 closes the gap, the finding is that the headroom was a guidance bug, not an RL opportunity.
- **Separate environment issue:** 1 of 11 cases is an unsafe designated target (map-vs-truth mismatch). This is about 2% of episodes at most and affects every policy equally.

### E026, 2026-10-15: Found a navigation-filter flaw (altitude lock-out after large terrain fixes); partial correction of E024 and E025
Tracing why the new landing sequence (`LandingGuidance`, "baseline v2", added to `trnrl/guidance.py`) still timed out at 600 m initial error, I printed the filter's altitude: in seed 20014 the lander estimated 10.6 m above the target while it was 92 m up, with horizontal error under 1 m. Time series (baseline guidance, same seed): altitude error grows to -82 m while the horizontal estimate is still 520 m wrong (the altimeter is fitted to the map under the wrong position), the first terrain fix then jumps the horizontal estimate and the error stays at -82 m for the rest of the episode.
- **Cause (code reading plus this trace):** the altimeter update is chi-square gated (1 dof, 10.8) with a small altitude variance; after a horizontal jump the height under the map changes by up to hundreds of metres, the residual exceeds the gate, and every later altimeter update is rejected. The altitude error never clears. It is a flaw in my nav filter, not in the terrain data or the guidance.
- **Option `nav.alt_sync` (default off, so earlier results reproduce), flag `--alt_sync` in the window scripts:** after an accepted TRN fix, add (map height change)^2 to the altitude variance so the gated update can pull the altitude back. A first version that shifted the altitude state by the map-height change made things worse (22/40 precise at sigma 600 vs 33/40; altitude errors up to 355 m), because the old altitude is not always consistent with the old position, so it was discarded.
- **Measured, site23, baseline guidance, widened search, paired seeds 10000+:**

| Setting | Precise without / with alt_sync | nav_ok_hazard (good horizontal nav, bad touchdown) | zero_fix (ambiguous) |
|---|---|---|---|
| sigma 120 (n=100) | 95% / 95% | 4 / 4 | 0 / 0 |
| sigma 600 (n=60) | 72% / 78% | 3 / 1 | 10 / 10 |
| sigma 1000 (n=60) | 85% / 92% | 5 / 1 | 1 / 1 |

  On the separate tuning seeds 20000+ (n=40): sigma 600 33/40 to 35/40, sigma 1000 34/40 with sync (no matched baseline run); remaining failures there are zero_fix, false_fix and one timeout.
- **Measured:** with the fix, the good-navigation, bad-touchdown failures nearly disappear at large initial error (3 to 1, 5 to 1 per 60); the ambiguity (zero_fix) failures at sigma 600 are unchanged (10 / 10); at sigma 120 nothing changes. The precise-rate gains (+6 and +7 points) are about one standard error each (SE about 6 points at n=60); the drop in type B counts is the clearer signal.
- **Correction to E025:** the failures I read as guidance deficiencies (hard landing at the target, loitering, crash short of the target) were at least partly caused by the altitude lock-out. The interpretation in E025 ("a deficiency of the hand-written guidance") is withdrawn pending a re-run of guidance variants with `--alt_sync`. E024 stays valid as a count of outcomes under the old filter; its "type B is a pure guidance problem" statement is withdrawn.
- **Baseline v2 (`LandingGuidance`):** first version approached too slowly (a bug in my gating condition, fixed); without alt_sync it ties the old baseline (33/40 vs 33/40 at sigma 600, tuning seeds, many identical failures), as expected if the cause was in the filter. Not yet run with alt_sync.
- **Hypothesis, not tested:** with alt_sync the remaining failures at sigma 600 are mainly ambiguity (no fix) and wrong accepted fixes, i.e. navigation, and the hold rule (which did not help in E024) is the only existing lever for the ambiguity case.
- **Limits:** one tile, 40 to 100 episodes per cell, two seed sets, filter fix is a heuristic (covariance inflation), not a re-derived filter; effect on Site 04, Haworth and the earlier PPO runs not measured (at sigma 120 no change on site23, so I expect little effect there).

### E027, 2026-10-16: Ceiling experiment. What would the baseline achieve with perfect navigation?
`scripts/diagnose_failures.py <tile> 600 100 oracle_<tile> baseline --oracle` (new env flag `oracle_nav`: guidance sees the true state; TRN off for speed). Same baseline guidance, same 100 episodes (seeds 10000+), 600 m initial error irrelevant under oracle.

| Tile | Precise with perfect navigation | Failures |
|---|---|---|
| site23 | 96 / 100 | 4 unsafe_site, landing error 0 m |
| site04 | 99 / 100 | 1 unsafe_site, landing error 0 m |

- **Measured:** with perfect navigation the baseline lands on the target to within 1 m every time; every failure (5 of 200) is an unsafe touchdown exactly at the designated target. The designated target is chosen as safe on the 20 m onboard map but is not safe on the 10 m truth (same artefact as seed 10035 in E025). The ceiling for any policy that only controls how it flies to the given target is therefore 96% on site23 and 99% on site04; it can only be exceeded by choosing a different, truly safe site (hazard detection and retargeting).
- **Measured, site23, paired seeds 10000 to 10059:** the fixed baseline (`--alt_sync`) fails 13 episodes at sigma 600; the oracle also fails on 1 of those (the target artefact), i.e. 12 of 13 are caused by navigation. At sigma 1000: 5 failures, 4 of them navigation. At sigma 120 the fixed baseline (95%) is already at the ceiling (96%).
- **Interpretation (measured gaps, untested mechanisms):** at default initial error there is no headroom for any flying policy. At 600 m initial error the headroom is 78% to 96% and is a navigation-information gap (no fix, wrong fix), which a policy can only close by getting more or better information, e.g. by choosing where to fly so terrain matching works. Separately there is a 1 to 4% gap from the unsafe-target artefact, closable only by site retargeting.
- **Limits:** one tile each; oracle removes all navigation error including vertical; ceiling depends on this environment's target selection and the 10 m truth safety criterion.

### E028, 2026-10-16: Active perception test. Does flying toward better-textured ground fix the no-fix failures?
`DetourGuidance` in `trnrl/guidance.py`, run through `scripts/diagnose_failures.py data/site23.npy 600 60 detour_600 detour --alt_sync` (60 episodes, seeds 10000 to 10059, paired with the fixed baseline `s23_600_sync`). Rule (hand-written, no learning): if the horizontal 1-sigma is above 60 m, no fix for 12 s and altitude above 400 m, fly one lateral detour (up to 600 m, 15 m/s) toward the cell with the highest onboard-map relief (std of height over 300 m, `Site.map_tex`), then resume the normal descent. Only one detour per episode.

| | Precise | no-fix failures (zero_fix) | false_fix | nav_ok_hazard |
|---|---|---|---|---|
| fixed baseline (alt_sync) | 47 / 60 | 10 | 2 | 1 |
| + hand-written detour | 47 / 60 | 10 | 2 | 1 |

- **Measured:** same total, but not the same episodes. The detour turned 3 of the 10 no-fix episodes (seeds 10010, 10017, 10022) into precise landings (20 to 26 terrain fixes each, 1 to 5 m error), and turned 3 precise episodes into no-fix failures (seeds 10007, 10041, 10055). In those 3 the baseline's first fix had come late (25 to 27 s) and the detour moved the lander off the ground track that would have produced it.
- **Measured:** 53 of 60 episodes have the same outcome and landing error within 5 m.
- **Interpretation (hypothesis for the mechanism, measured counts):** seeking a different ground track does produce fixes in a minority of the ambiguous cases, so active perception can add information; the relief-std texture proxy does not predict where matching works well enough to be right more often than wrong, and the rule cannot tell which episodes need it. Choosing detour or no detour per episode with hindsight would give 50 / 60 at best for this rule (+5 points), which is within noise at n = 60 and is an upper bound for this particular rule, not for the idea.
- **Next:** replace the relief proxy with a direct matchability map computed from the onboard map (self-similarity ratio of the NCC surface within the search radius), and let a learned policy decide whether and where to detour. Success criterion fixed in advance: more than +8 points precise at sigma 600 on held-out seeds relative to the fixed baseline, with no loss at sigma 120.
- **Limits:** one tile, 60 episodes, one rule, one detour per episode; seeds not separated into tuning and report sets for this run (no tuning was done on it).

### E029, 2026-10-16: Why the hand-written detour failed, a stronger classical baseline (matching patch size), and the learned hold/detour decision
Four measurements, in order. Code: `trnrl/matchability.py`, `trnrl/detour.py` (`DetourGuidance2`, `DetourEnv`), `scripts/train_detour.py`, `scripts/eval_detour.py`, `--patch_n` flag.
1. **First fix is the bottleneck (site23, sigma 600, alt_sync, 40 episodes, 1231 fix attempts).** Attempts made while the filter's pre-attempt horizontal sigma was under 60 m succeeded 863 / 863 times; attempts at sigma above 450 m succeeded 8% of the time (368 attempts, all the rest `rejected_ambiguous`); no attempts in between. Among the high-sigma attempts, the new matchability score (`matchability_map`: second-peak separation outside the main lobe times a lobe-sharpness term, from the onboard map only) predicts success with AUC 0.66; the relief-std texture score used by the E028 rule has AUC 0.33, i.e. it points the wrong way. **Measured.** (An earlier version of this analysis that used the post-update sigma was leaking the label and is not reported.)
2. **Matching patch size (site23, sigma 600, alt_sync, 60 paired episodes, baseline guidance).**

| patch (cells) | Precise | zero_fix | false_fix | nav_ok_hazard |
|---|---|---|---|---|
| 15 (default) | 47 / 60 (78%) | 10 | 2 | 1 |
| 23 | 53 / 60 (88%) | 6 | 0 | 1 |
| 31 | 54 / 60 (90%) | 5 | 0 | 1 |
| perfect navigation (E027, same seeds) | 59 / 60 | | | |

   A larger matching footprint is a classical fix that recovers most of the navigation gap at large initial error; it assumes a sensor that can cover 460 to 620 m of ground at 20 m spacing from high altitude (not checked against a real instrument). **Measured; sensor realism is an assumption.** With patch 31 the remaining no-fix episodes are about 8% on site23 and about 1% on site04.
3. **Training feasibility on real tiles.** With patch 31 at sigma 600 the learned-detour environment found no episode on site04 that needed a decision in 400 consecutive resets, so there is almost nothing for a decision policy to learn from there; on site23 roughly 1 episode in 12. **Measured (site04 observation), estimated (site23 rate).**
4. **Learned hold/detour decision on the synthetic low-relief benchmark `mixed_hard+dec`** (held-out test sites, 200 episodes, seeds 10000+, same seeds for every policy; SE about 2.5 points; alt_sync off as in E013):

| Policy | Precise | no-fix episodes |
|---|---|---|
| baseline | 83.0% | 32 |
| hand-written hold (70 s) | 89.0% | 21 |
| always hover 25 s per decision (up to 3) | 87.5% | 22 |
| hand-written detour toward best matchability | 75.5% (43 timeouts) | 9 |
| PPO from scratch (1500 decisions, about 600 episodes) | 81.5% | 35 |
| PPO warm-started from always-hover, seed 0 (6000 decisions, 2200 episodes, 41 min) | 87.5% | 22 |
| same, seed 1 | 87.5% | 22 |
| perfect navigation (ceiling, 200 episodes) | 99.5% | |

   **Measured:** both warm-started runs end with exactly the always-hover outcomes; PPO found nothing better than hovering with this action set (hover or lateral detour toward 16 candidate points, 31 features including the matchability map). Training reward curves are flat at 0.78 to 0.86 over decision episodes.
- **Interpretation (hypotheses):** (a) the 10-point gap between hold (87.5 to 89.0%) and the ceiling (99.5%) on this benchmark is made of episodes where no terrain fix ever arrives; hovering does not create one and the lateral detours did not either (they cost time). (b) On real tiles the dominant lever turned out to be the matching footprint and the filter, not the flight policy.
- **Limits:** 2 PPO seeds; one policy class and reward; matchability validated on one tile; the real-tile numbers use alt_sync, the synthetic ones do not; one sensor-footprint model.
- **What this does not show:** it does not show that no RL policy could beat these baselines; it shows that this action set, this reward, and this much training did not.

## 5. Open questions
1. ~~How much of the baseline's tail comes from filter overconfidence?~~ Partly answered by E011 (decoupling helps in `mixed_hard`, hurts with no TRN). Still open: a calibrated filter that keeps the altimeter's real positioning information.
2. Does a policy with covariance in its observation learn anything a hand-written rule cannot? The hold rule (E013) is the bar: 89.0% (decoupled) / 80.5% (coupled) on `mixed_hard`. Three experiments (E014, E018, E020) say no for the residual PPO as built; `cov` was worse than `nocov` each time. Open: whether a different action space (learned hold/detour) changes that.
2b. Is the detour-toward-texture lever real? On low-relief sites without blackout 18% are zero-fix; a rule or policy that flies toward textured crater rims (visible in the texture crop) might recover some. Untested.
2c. ~~Does retraining PPO on the decoupled filter change the cov vs nocov result?~~ Answered by E018: no, cov got worse (one seed).
3. Does the policy exploit simulator quirks (e.g., the 1 s control step, perfect thrust-direction tracking)? Plan: evaluate on held-out sites and with perturbed dynamics (thrust +/-5%, 1 to 2 s actuator lag).
4. Real terrain: do conclusions survive? Partly answered (E016, E020): on two rugged real tiles the baseline is already at 95 to 98% precise and the hold rule changes nothing. Haworth (E021) also shows no headroom with the default setup; the headroom on real tiles appears only under stress (300 m initial error, E022). Still open: truly flat low-relief real terrain (none of the three sites).
5. (new, E020) The baseline's remaining losses on Site 23 are `unsafe_site` and hard landings, not navigation misses. Are they avoidable with the 20 m onboard map, or invisible at that resolution?
6. (new, E019/E020) Why does PPO degrade late in training (final cov 85.5%)? Untested: policy-std or value drift, covariance-feature overfit, reward trading safety for fuel.

## 6. Plan and checklist
- [x] Terrain module, hazard/texture maps
- [x] EKF + TRN with gating and calibrated covariance
- [x] Gymnasium env, polynomial baseline, evaluation harness, NEES logging
- [x] Baseline in nominal / no-TRN / low-relief regimes
- [x] Unit tests (17 passing)
- [x] Hard-scenario benchmark (mixture of regimes, held-out sites)
- [x] Sandbox + random-window training pipeline built (E017) and run on real terrain (E019, E020): no RL gain
- [~] PPO with vs. without covariance: seed 0 done on synthetic (E014, E018) and real windows (E020), all negative; need 3+ seeds and a redesigned action space before concluding
- [x] Stronger hand-written baseline (covariance-triggered hold, E013)
- [ ] Detour-toward-texture rule/policy
- [x] PPO on decoupled filter (E018): negative
- [ ] Learned hold/detour decisions with more authority (the next RL experiment)
- [~] Altimeter overconfidence: decoupled mode added and tested (E011); calibrated-coupling variant not built
- [~] Held-out sites: done for real terrain (train Site 04, test Site 23, E020) and synthetic (disjoint seeds); perturbed dynamics (thrust error, actuator lag) not done
- [ ] 3+ seeds per RL variant, with confidence intervals across seeds
- [ ] Hazard-avoidance retargeting experiment (E020 open question 5)
- [x] Haworth run (E021): baseline 96 to 99.5%, no headroom; no flat low-relief real tile found yet
- [x] Check the big_init result (E023): it was the 20-cell search window, fixed with `--init_pos_h` (auto-widens search)
- [x] Phase 1: failure diagnosis (E024): ambiguity no-fix (type A) and good-navigation bad-touchdown (type B); hold rule does not help
- [x] Phase 2 step 1: inspect type B (E025): guidance deficiencies with fuel left, one unsafe-target artefact
- [x] Phase 2 step 2a: found and fixed (optionally) the altitude lock-out in the nav filter (E026)
- [x] Phase 2 step 2b: ceiling (E027), detour test (E028), matchability validation, patch-size fix and learned hold/detour (E029); no RL gain found
- [ ] Re-measure Site 04, Haworth and PPO baselines with --alt_sync (flag is off in all earlier results)
- [ ] Phase 3: seeds, ablations, validate baseline vs published numbers
- [ ] Phase 4: write-up
- [~] Real-tile stress tests (E022): 300 m initial error done on two tiles; thrust error / actuator lag / noisier sensors not done
- [x] Real LOLA tile: two PGDA 8 km tiles run (E016, E019, E020)
- [~] 3D visualisation: static 3D terrain figure (E015) and an interactive local 3D sandbox with safe-ground and terrain-matching overlays and uncertainty rings (E017); covariance ellipsoids (3D) and a posterior-probability heat map not done
- [x] Explainer figures + plain-language page for non-technical audiences (E015)

## 7. Reproducibility
```
pip install -r requirements.txt     # numpy scipy gymnasium stable-baselines3 torch matplotlib pillow rasterio pytest
python scripts/evaluate.py baseline 100 baseline_synthetic_trn nominal
python scripts/evaluate.py baseline 100 baseline_no_trn no_trn
python scripts/evaluate.py baseline 100 baseline_low_relief low_relief
python scripts/evaluate.py baseline 200 baseline_mixed_hard mixed_hard
python scripts/sweep_trn_ambiguity.py 150
python scripts/evaluate.py baseline 200 baseline_mixed_hard_dec200 mixed_hard+dec
python scripts/train_ppo.py cov 0 600000 mixed_hard        # and: nocov  (~28 min each on 1 core)
python scripts/evaluate.py ppo:results/ppo_cov_mixed_hard_s0:cov 200 ppo_cov_s0_test mixed_hard
python scripts/evaluate.py ppo:results/ppo_nocov_mixed_hard_s0:nocov 200 ppo_nocov_s0_test mixed_hard
python scripts/evaluate.py wait:70 200 wait70_test_mixed_hard+dec mixed_hard+dec test
python scripts/evaluate.py wait:<S> 200 sweep_wait<S>_train mixed_hard+dec train      # S in 0 30 50 70 80 90
python scripts/train_ppo.py cov 0 600000 mixed_hard+dec ; python scripts/train_ppo.py nocov 0 600000 mixed_hard+dec   # E018 (tags ppo_*_dec_s0 via 5th arg)
python scripts/evaluate.py ppo:results/ppo_nocov_dec_s0:nocov 200 ppo_nocov_dec_s0_test mixed_hard+dec test
python scripts/evaluate.py baseline 100 real_base_test real+dec test     # E016 (needs data/site04.npy, data/site23.npy from scripts/crop_dem.py)
python scripts/crop_dem.py Site04_final_adj_5mpp_surf.tif --size_km 8 --out data/site04.npy   # on the user's machine
python scripts/train_ppo_windows.py $TRAIN cov 0 600000 48 ppo_win_cov_s0 --frac 1.0     # E019/E020, then eval_windows.py on $TEST (see SANDBOX.md)
python scripts/make_figures.py          # E015 figures
python sandbox/server.py --dem $TEST [--model ... --variant ...]       # E017 sandbox
python -m pytest tests                  # 20 tests
```
Evaluation seeds 10000 to 10099, site seeds 1000 to 1007. All randomness flows from `np.random.default_rng` seeded in `env.reset(seed=...)`.

## E030 — Paper mock + one-command experiment suite (infrastructure, no new results)
- Built `scripts/run_paper.sh` (parts A–E, T) and `scripts/paper_tables.py`; all parts smoke-tested with QUICK=1 (N=2–3, not real data; deleted).
- Mock paper drafted in Claude Docs: "Where Does Learned Guidance Help? Real-Terrain Lunar Landing Benchmark"; all numbers [TBD] until the full run.
- Hypothesis (unchanged): baseline near perfect-nav ceiling in default regime; headroom only at large initial error; RL gives no gain. To be confirmed with 3 seeds + CIs.
- Checklist: run suite (pending, user), validate baseline vs published numbers (pending), hazard-retargeting (not built).

## E031 — Realism layer (map registration error, correlated map noise, FOV-based patch) + suite rework
Motivation (measured): E030 comparison — our median landing miss was 1.3–1.7 m vs flown 5–9 m (Mars 2020 ~5 m, post-landing est. 8.5 m or better; LVS map-relative <40 m). Idealised onboard map (derived from truth, no registration error) is the likely cause (HYPOTHESIS).
Code (new knobs, all default OFF so E001–E030 stay reproducible): nav.reg_offset_m (fixed per-episode map-to-world offset, 1σ/axis, biases every TRN fix), env.map_noise_m / map_noise_corr_m (correlated vertical error added to the onboard map, also affects target selection), nav.fov_deg (patch side = 2·agl·tan(fov/2) in map cells, clipped 7..41; replaces fixed patch_n). Flags: --fov --reg_offset --map_noise --map_corr, preset --realistic (fov 40°, offset 6 m, noise 1 m — ASSUMED values, not published; swept in part R). NCC search chunked (the wide-search + big-patch case was OOM-killed).
Measured (smoke only): site23 σ120 with --realistic: median miss ≈ 6–13 m on 12 episodes (was ~1 m) — now in the flown 5–9 m range. mixed_hard+dec N=6: 0.4–9.6 m vs 0.9–2.5 m legacy (excl. one 250–360 m failure in both).
Suite: parts A/B/C now run under --realistic (fov60 replaces patch31 as the "larger sensor" variant), new part R (one-knob sensitivity + RL robustness under realism), part L (legacy conditions for comparison). Tables now include median/p90 miss.
Open: values of the three realism knobs are assumptions; justify or sweep (part R). Full run not yet executed.


### E032, 2026-10-17: Final suite results with the realism layer (parts A, B, D, E, L, R measured; part C table pending)
Source: results/paper_tables.md from `scripts/run_paper.sh` (n=200 for A, D, E, R-dec; n=100 for B, L and the R sensitivity sweeps; realism preset = fov 40 deg, 6 m offset/axis, 1 m map noise unless stated).
**Measured**
- A (default 120 m, n=200): baseline precise 96.5 / 99.0 / 98.5 / 95.5% (Site04 / Site23 / Haworth test / Haworth train); perfect navigation 98.5 / 97.5 / 98.0 / 97.5%; median miss 7.4 to 8.4 m, p90 13 to 15 m. alt_sync column identical to non-fixed at default error. Hold rule within noise of baseline.
- B (n=100): narrow 20-cell window 81% (sigma 300) then 24/25% (600) and 12% (1000) on both tiles. Wide window: Site04 97/99/99%, Site23 99/89/94%. +alt_sync: Site23 99/90/96%. +fov60: Site23 99/92/97%. +hold: Site23 99/91/96%. At Site23 sigma 600 nearly every failure is a zero-fix episode (7 to 9 of 7 to 10).
- R sensitivity (Site23, alt_sync): sigma 600 precise by fov 20/30/40/60 deg = 81/86/90/92%, zero-fix 17/12/9/7; sigma 120 flat 97 to 99%. Offset 0/3/6/12 m: median miss 3.8/5.5/8.4/14.8 m, precise unchanged (98 to 100% at 120, 89 to 91% at 600). Map noise 0 to 3 m: median miss 7.6 to 13.8 m, precise 96 to 99% (120) and 88 to 90% (600).
- D (idealised map, n=200): PPO nocov seeds 0/1/2 = 82.5/77.0/79.0%, cov = 59.0/66.5/79.5% vs baseline 83.0%, hold 89.0%. Haworth PPO: s1 best 98.5 / final 87.5% (21 hard landings); s2 best 99.5 / final 98.0%. No run beats baseline beyond SE. Earlier claim "cov worse than nocov in every run" is FALSE with 3 seeds (seed 2: cov 79.5 vs nocov 79.0).
- E / R-dec (mixed_hard+dec, n=200): baseline 83.0 nominal, 73.0 realism (zero-fix 32 -> 49); hold 89.0 / 78.0; hand detour 75.5 / 76.5 (43 and 31 timeouts); always-hover 87.5; learned seeds 0, 1, 2 identical to always-hover (87.5 nominal, 77.5 realism, identical p90); oracle 99.5.
- L (n=100, idealised map): Site23 sigma 120 legacy 95 / sync31 96%; sigma 600 legacy 73 / sync31 88%.
**Interpretation (hypotheses unless stated)**
- The three learned-decision "seeds" are one policy: measured fact is identical outcomes; cause (hover-logit bias 4.0, lr 1e-4, 6000 decisions, greedy action never flips) read from the code, not tested by a bias sweep. Do not report as 3 independent results.
- Baseline above the ceiling on Site23 (99.0 vs 97.5, within SE): HYPOTHESIS that the 6 m offset moves some touchdowns off an unsafe designated cell. Untested.
- alt_sync matters less under the fov-40 patch (89 vs 90, 94 vs 96): HYPOTHESIS that earlier first fixes reduce lock-out. Untested.
- Site23 sigma 1000 (94 to 97%) above sigma 600 (89 to 92%): read as noise (SE 2 to 3); needs more episodes.
- Hold rule +5 to +6 points on mixed_hard (both conditions) is about 1.5 to 2 SE: suggestive, not established; paired CI pending.
**Housekeeping:** `paper_tables.py` now prints a part-C category table (the diagnosis json has no `summary`); rerun `PART=T bash scripts/run_paper.sh`. Mock paper updated with these numbers (abstract, 5.1, 5.2, 5.5 to 5.7, discussion, limits).


### E033, 2026-10-17: Decision-layer headroom, and a one-line rule that captures it (supersedes the E029 reading of the learned detour)
Setup: `scripts/decision_headroom.py` tries each first-decision action (0 none, 1..8 detour 400 m, 9..16 detour 800 m, 17 hover), then no further detours, on the same episodes; `scripts/fit_decision_policy.py` fits a chooser on training-site logs and evaluates exactly on test logs (all actions' outcomes known per episode). mixed_hard+dec, test n=200 (110 reach a decision point), train n=400 (217 with a decision point), seeds 10000+i (test) / 50000+i (train), separate sites.
**Measured**
- Nominal test: none 81.5%, hover 86.0%, mean over 16 single detours 89.8% (86.0 to 92.5%), hindsight best of all 98.5%. Realism layer (headroom only): none 73.5, hover 76.0, mean detour 79.3 (75.5 to 83.5), hindsight 99.5.
- Hand matchability rule at the FIRST decision only: 90.0% (zero-fix 17), vs 75.5% with up to 3 decisions (43 timeouts): the E028 failure was repeated decisions, not detouring.
- Matchability rule 90.0% vs random detour 89.7%: no direction information in the matchability map as used.
- Detour options binned by alignment with the direction to the target (held-out): alignment -1..-0.5 (away) 89.1% precise (n=586), 0.5..1 (toward) 75.9% (n=586).
- One-line rule "detour 800 m directly away from the target": 93.5% test, 92.7% train (nothing fitted); at 400 m: 91.0% test.
- Conditional-logit chooser (shared weights, 18 intercepts, C=0.1 by 5-fold CV on train): 94.5% test. Paired bootstrap vs none +13.0 (CI +8.0..+18.0), vs hover +8.5 (+5.0..+12.5), vs best fixed action +5.0 (+1.5..+8.5), vs matchability rule +4.5 (+1.0..+8.5), vs random detour +4.8 (+2.4..+7.3). 30 bootstrap refits of the training set: 92.5 to 95.5%, mean 93.7%. Ablation: without alignment features 89.5% (= best fixed action); without matchability 94.0%.
- Learned chooser minus one-line rule: +1.0 point at n=200: not significant (no test run; difference is 2 episodes).
**Interpretation (hypotheses, untested)**
- Why away from the target helps: more time and a second viewing geometry for the first terrain fix. No experiment separates the two.
- Hindsight 98.5% is an optimistic cap (best of 19 tries per episode; noise stream shared across actions).
- E029's statement that the learned detour 'equals hover' reflects the warm start (bias 4.0 on the hover logit, lr 1e-4, 6000 decisions), not that detouring cannot help.
**Scope limits:** synthetic low-relief benchmark, nominal conditions for the fit; supervised learning on full-information logs, not RL; real tiles and realism-layer fit not done.
**Housekeeping:** new `PART=H` in run_paper.sh; paper_tables.py prints H and H2 tables; mock paper section 5.8 added and abstract/contributions/discussion revised.

### E034, 2026-10-17: Part C under realism, and realism-layer decision headroom on both splits (realism-layer chooser fit produced in E035)
**Measured**
- C (Site 23, 60 episodes each, realism layer): sigma 600: legacy-window 53 precise (zero_fix 5, nav_ok_hazard 2); +alt_sync 54 (zero_fix 5, nav_ok_hazard 1); +fov60 54 (zero_fix 5, nav_ok_hazard 1). sigma 1000: 57 (nav_ok_hazard 1, nav_bad 1, other 1), 59 (nav_ok_hazard 1), 59 (nav_ok_hazard 1). No zero-fix episodes at 1000 m. Consistent with B (89/90/92% at 600 m).
- H realism-layer headroom, train n=400 (165 with a decision point): none 73.0, hover 70.8, mean single detour 77.2, best single 80.5, hindsight 95.8. Test n=200: none 73.5, hover 76.0, mean detour 79.3, best 83.5, hindsight 99.5. Nominal train n=400: none 84.5, hover 86.0, mean detour 89.8, best 93.5, hindsight 99.5.
- Hover vs none flips sign across realism splits (+2.5 test, -2.2 train, SE about 2 to 3): not reliable. Mean single detour beats none by +5.8 (test) and +4.2 (train).
**Not done:** `P_H_fit_realistic` is absent from the table: the fit step either failed or did not run after the realism jobs finished. Command to rerun and read the error: `python scripts/fit_decision_policy.py results/P_H_train_realistic.json results/P_H_test_realistic.json P_H_fit_realistic`.
**Housekeeping:** mock paper 5.2 and 5.8 and the limits list updated.

### E035, 2026-10-17: Decision chooser fitted under the realism layer (closes the E034 gap)
Setup: `fit_decision_policy.py` on P_H_train_realistic (n=400, 165 with a decision point) and P_H_test_realistic (n=200, 88 with a decision point). Requires scikit-learn (was missing; installed).
**Measured (test, precise = within 50 m, n=200)**
- none 73.5%, hover 76.0%, random detour 79.3%, best fixed action (chosen on train) 81.0%, matchability rule 81.5%, conditional logit 83.5% (C=1.0), away-800 rule 84.5%, away-400 rule 81.5%, hindsight best 99.5%.
- Train-set 5-fold CV score of the chooser on decision episodes only: 0.56 to 0.58 (nominal: 0.87 to 0.89).
- Learned chooser vs none +10.0, vs best fixed +2.5, vs random detour +4.2 (point estimates; paired bootstrap not run). Learned chooser is 1.0 below the one-line away-800 rule.
**Interpretation (hypotheses, untested)**
- Under realism the learned chooser has no demonstrated advantage over the one-line rule; the decision-time features predict little of the per-episode variation (CV 0.56 to 0.58 vs hindsight 99.5%), possibly because outcomes are dominated by noise draws that cannot be seen at decision time. Not tested.
- Detouring away from the target helps in both settings (nominal 93.5 vs 81.5 none; realism 84.5 vs 73.5 none): the robust finding. The learned layer adds nothing measurable beyond it.
**Scope limits:** only 88 test episodes reach a decision point, so differences of 1 to 3 points are 2 to 6 episodes; synthetic benchmark; realism knob values assumed.
**Housekeeping:** mock paper section 5.8 table, caveats, abstract and discussion updated; `paper_tables.py` H2 does not yet show away_800_rule / away_400_rule columns.

### E036, 2026-10-17: Paired bootstrap for the realism-layer chooser
Setup: `scripts/bootstrap_decision.py` (new): 2000 paired resamples of the 200 test episodes, chooser fixed (C=1.0); plus 100 refits on resampled train decision episodes. Output results/P_H_boot_realistic.json.
**Measured (percentage points, 95% CI)**
- chooser minus none +10.0 (4.5, 15.5); minus hover +7.5 (2.0, 13.0); minus random detour +4.2 (0.4, 8.1); minus best fixed +2.5 (-2.5, 7.5); minus matchability rule +2.0 (-3.5, 7.5); minus away-800 rule -1.0 (-6.0, 4.0).
- Refit range over train resamples: 79.5 to 87.5%, mean 84.0%.
**Interpretation:** learned chooser is better than no detour, hover and random detour, but not distinguishable from the best fixed action or the rules at this sample size (hypothesis that more decision episodes would separate them: untested).
**Housekeeping:** mock paper 5.8 updated. Nominal-case bootstrap was in E033.

### E037, 2026-10-17: Paired CIs for the main tables; Haworth checkpoint selection confirmed
Setup: `scripts/paired_ci.py` (new) pairs episodes by seed across result jsons (precise = success and landing error <= 50 m; verified equal to the summaries for every file), 5000-resample paired bootstrap and exact sign test; output results/paired_ci.md, also appended to paper_tables.md (PART=T now runs it first). 70+ comparisons over A, B, D, E, R.
**Measured (95% paired CI, points)**
- A: baseline minus perfect navigation: Site 04 -2.0 (-4.5, 0.0), Site 23 +1.5 (-1.0, +4.0), Haworth test +0.5 (-1.0, +2.5). alt_sync window fix vs legacy: 0 discordant episodes on all tiles at default settings.
- B: wide vs narrow window +64 to +87 pts at 600/1000 m (CIs exclude 0 by far); alt_sync vs wide: Site 23 600 m +1.0 (0.0, +3.0), 1000 m +2.0 (0.0, +5.0), Site 04 0.0; sync+fov60 vs sync Site 23 600 m +2.0 (0.0, +5.0); hold vs baseline within +-1 pt everywhere.
- R (Site 23, 600 m, n=100): fov60 minus fov20 +11.0 (+5.0, +17.0), sign p=0.001; fov20 vs preset -9.0 (-15.0, -4.0); fov30 vs preset -4.0 (-8.0, -1.0), p=0.125; fov60 vs preset +2.0 (0.0, +5.0). Offset 0/3/12 and map noise 0/2/3 vs preset: all intervals include 0.
- E/D synthetic (n=200): hold minus baseline +6.0 (+2.0, +10.0), p=0.008; under realism +5.0 (+0.5, +9.5), p=0.064. Hand detour vs baseline -7.5 (-14.0, -1.5) nominal, +3.5 (-2.5, +9.5) realism. Warm-started learned seeds = hover (identical). Learned minus baseline +4.5 (+1.0, +8.5) = hover's gain; learned minus hold -1.5 (-4.0, +1.0).
- PPO residual vs baseline: no-cov s0/s1/s2 -0.5 / -6.0 (-11.0, -1.5) / -4.0 (-8.5, +0.5); cov -24.0 (-31.5, -16.5) / -16.5 (-23.0, -10.0) / -3.5 (-8.0, +0.5). Haworth best: s1 0.0 (-2.0, +2.0), s2 +1.0 (0.0, +2.5); final: s1 -11.0 (-15.5, -6.5), s2 -0.5 (-2.5, +1.5).
**Checked in code (train_ppo_windows.py):** `_best` = highest rolling precision logged during training (stochastic policy, training windows only); test strip never used for selection. Optimistic and noisy, but not leaked.
**Not changed / caveats:** Paired CIs treat the 200 (or 100) seeds as the sample; training-seed variance is separate (3 seeds, learned-decision seeds identical). P_E baseline (83.0%) differs from the H 'none' rate (81.5%) because H logs are a separate run with different random streams; both reported as is.
**Housekeeping:** paper_tables.py H2 table now shows away-800 and away-400 rule columns; paper Statistics paragraph, 5.1, 5.2, 5.4, 5.6, 5.7 updated.

### E038, 2026-10-17: Site 23 extended runs (400 episodes, same seeds; part X) and the baseline-vs-reference diagnosis
Setup: `PART=X NX=400` (10 jobs, ~5.5 h on Ryzen 6600H; the sigma1000 sync_hold job was killed after 137 CPU-min, so that cell is missing). First 100 seeds equal part B, so results are paired with B. Estimated runtime in advance (30 to 60 min) was wrong: wide/hold runs at large sigma are slow.
**Measured (precise <= 50 m, n=400; paired 95% CI in points)**
- sigma 600: wide 86.0, sync 88.0, sync+fov60 89.2, fov30 85.5, fov20 76.8, hold 88.0. sigma 1000: wide 90.0, sync 92.2, sync+fov60 95.5.
- alt_sync vs wide: +2.0 (+0.8, +3.5) at 600 m, +2.2 (+1.0, +3.8) at 1000 m (8/0 and 9/0 discordant).
- fov at 600 m: 20 vs 40 -11.2 (-14.8, -8.0); 30 vs 40 -2.5 (-4.8, -0.2), p=0.052; 60 vs 40 +1.2 (-0.2, +2.8), p=0.18; 60 vs 20 +12.5 (+9.0, +16.2). fov 60 vs 40 at 1000 m +3.2 (+1.5, +5.2), p=0.001.
- hold vs baseline at 600 m: +0.0 (-1.8, +1.8).
- sigma 1000 minus sigma 600 (sync): +4.2 (+1.5, +7.2), p=0.006: the 1000 m cell is easier, not noise. Zero-fix episodes 38 (600 m) vs 16 (1000 m). Failures at 600 m are mostly unsafe_site (39 of 48) together with zero fixes.
- Site 23 baseline vs perfect navigation (n=200, 7 discordant episodes): in 5 the oracle lands within 0.1 m of the designated target and that target is unsafe on the truth map (baseline misses by 7 to 14 m and lands on a safe cell); in 2 the baseline fails and the oracle succeeds. Measured explanation for baseline above the reference; "ceiling" renamed "perfect-navigation reference".
**Interpretation (hypotheses, untested):** why sigma1000 > sigma600 (guesses: interaction of initial offset, 30 m patch footprint and relief along the approach).
**Housekeeping:** mock paper 5.2, 5.3, 5.5 updated (the earlier 'noise' reading of sigma1000 > sigma600 was wrong and is replaced); reviewer-style remarks received and triaged (residual-scale ablation, independent-noise hindsight check, first-fix-time log added to the plan).

### E039, 2026-10-17: Mechanism scripts written (not yet run)
`scripts/mechanism.py` with three modes: `geom` (time-matched hover 25/60 s vs detours away/toward/perpendicular at 400/800 m, with time to next fix and fixes after the decision), `noise` (replays each decision episode R times per action with different measurement noise: per-episode success probabilities and a pure-luck cap = best of R replicates of the same action), `fix` (first-fix time, rejections before the first fix, altitude-state error at touchdown for legacy/fov40/fov60 x alt_sync off/on). Smoke-tested with tiny N only. Hypotheses to test: (1) away helps because of geometry, not time (the earlier alignment bins already hold time fixed: away 89.1% vs toward 75.9%); (2) the hindsight cap is largely luck; (3) altitude fix matters less under realism because the first fix comes earlier.

### E040, 2026-10-17: Mechanism check 1 (time vs direction), nominal mixed_hard+dec, 110 of 200 test episodes with a decision point
**Measured (precise on decision episodes; summary only, per-episode file not yet received so no paired CIs)**
- none 68.2; hover 25 s 76.4; hover 60 s 79.1; away 400/800 m 85.5/90.0; perpendicular 400/800 m 82.7/85.5; toward 400/800 m 68.2/75.5.
- Accepted fix after the decision: none 68.2% (identical to its precise rate), hover25 78.2, hover60 80.0, toward800 77.3, perp800 89.1, away800 92.7. Median total flight time: hover60 195 s vs away800 201 s, perp800 184 s.
- Mean accepted fixes after decision: none 7.1, hover60 9.5, away800 10.9.
**Interpretation:** time alone gives some gain (+8 to +11 points over none) but at matched flight time away is 11 points above hover 60 s and 14.5 above toward: direction matters. Why direction matters is untested (candidate: longer return leg gives more matching attempts at low altitude; not terrain matchability, which has no direction information).
**Pending:** M_geom.json for paired CIs; checks 2 (noise/luck) and 3 (first-fix timing, altitude fix).
**E040 addendum (paired CIs from M_geom.json, n=110 decision episodes, 95% bootstrap, points):** away800 - hover60 +10.9 (+5.5, +17.3), 13/1, p=0.002; away800 - toward800 +14.5 (+6.4, +22.7); away400 - toward400 +17.3 (+9.1, +26.4); hover60 - none +10.9 (+2.7, +19.1); hover25 - none +8.2 (+0.9, +15.5); away800 - perp800 +4.5 (-3.6, +12.7); toward800 - hover60 -3.6 (-10.0, +2.7). Direction effect holds at matched time; away vs perpendicular not separable.
