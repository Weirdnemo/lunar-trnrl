# Research Log: RL guidance for lunar powered descent with TRN

Working title: *Uncertainty-aware learned descent guidance under degraded terrain-relative navigation.*
Owner: Nimesh Chauhan.

Experiments only, in order. Entry IDs match the old log, so the gaps are just entries that weren't experiments. Numbers come from `scripts/` and `results/*.json` unless an entry says otherwise. One seed per variant unless noted, so "not supported" doesn't mean "refuted".

**Quick vocab**
- **Precise landing** = safe touchdown (vz <= 2 m/s, vh <= 1 m/s, slope <= 10 deg, roughness <= 0.6 m) within 50 m of the designated target. Main metric.
- **Baseline** = polynomial guidance down to a 30 m gate, then constant-rate vertical descent. **Hold rule** = hover while nav sigma is large (E013). **+dec** = decoupled altimeter (E011). **alt_sync** = altitude lock-out fix (E026).
- **mixed_hard** = half rough / half low-relief synthetic sites, 40% of episodes get a 30 to 70 s TRN blackout. Test sites are held out, episode seeds start at 10000.
- **sigma** = 1-sigma initial horizontal position error (default 120 m). **zero-fix** = episode where TRN never gave an accepted fix.

---

## Experiments

### Baseline benchmark (E006), 2026-10-14
100 episodes (200 for mixed_hard), seeds 10000+, 8 synthetic test sites.

| Condition | Safe | Precise | Median / p90 err | TRN fix rate | Zero-fix eps | Median NEES (dof 9) |
|---|---|---|---|---|---|---|
| nominal | 99% | **97%** | 1.1 / 1.8 m | 92% | 0 | 6.7 |
| TRN off | 93% | **42%** | 69 / 336 m | 0% | 100 | 69.4 |
| low_relief (4 m RMS, 8 craters) | 98% | **74%** | 4.2 / 204 m | 21% | 23 | 8.8 |
| mixed_hard (200 eps) | 94% | **75.5%** | 1.9 / 168 m | 53% | 42 | 8.1 |
| mixed_hard+dec (200 eps) | 99% | **83.0%** | 1.5 / 110 m | 58% | 32 | 7.4 |

- Nominal is saturated, nothing for RL to win there.
- Safe landing is a misleading headline: no-TRN is 93% safe but only 42% precise, because most synthetic terrain is safe (93 to 98%) so a lander 300 m off still lands fine. Hence precise landing.
- With TRN the filter is consistent (NEES 6.7 vs ~8.3 expected). Without it, NEES is 69 and 71% of episodes sit above 2x dof, so the filter is badly overconfident.
- Low relief: TRN accepts only ~21% of attempts. Median is fine, tail is bad (p90 204 m, 26% miss the 50 m mark).
- SE near 80% at n = 200 is ~2.8 pts, so gaps under ~5 pts between rows aren't reliable alone.

### E004, 2026-10-13: Synthetic terrain calibration
- Fractal (Hurst 0.85, 18 m RMS) plus power-law craters: median slope 3 to 5 deg, 94 to 98% of cells under 10 deg (4 seeds).
- Rougher settings (Hurst 0.5 to 0.7, relief 40 to 60 m): median slope 12 to 27 deg and zero safe targets. Dropped.
- Catch: this terrain is milder than real polar terrain.

### E005, 2026-10-14: Baseline bring-up, what broke
1. **Polynomial guidance hovered instead of landing.** Aiming 3 m above the target plus the time-to-go clamp gives ~0 net acceleration near zero speed. 11 of 12 smoke episodes timed out. Fix: two-phase guidance (polynomial to the gate, then constant-rate descent). Standard practice, so the baseline isn't a strawman.
2. **TRN search window capped at +/-10 cells (+/-200 m)** vs a 3-sigma of 360 m. Raised to 20 cells, seed 9 still failed, so the cap wasn't the (only) cause. Its separate effect wasn't isolated here (see E023).
3. **TRN ambiguity test rejected good matches** (the real cause of seed 9). It only excluded a 2-cell neighbourhood around the peak, but fractal terrain has a long correlation length, so the lobe's shoulder looked like a rival peak. Replaced with a lobe-based test. Sweep of 6 threshold pairs on 150 first-fix attempts (`scripts/sweep_trn_ambiguity.py`):
   - Chosen (0.10, 0.03): 79% accepted, 0% of accepted fixes wrong by > 60 m, median fix error 5.0 m.
   - Stricter margin (0.10 to 0.10): acceptance fell to 45%, no accuracy gain.
   - 20% of first attempts fail NCC >= 0.7 under every setting.
   - Catch: thresholds were tuned on the same synthetic terrain family used for eval. Re-tune on training sites before reporting.

### E007, 2026-10-14: Filter consistency
- TRN off: median NEES 69 vs ~8.3 expected. In one denied smoke episode sigma dropped from 120 m to ~19 m while true horizontal error grew to ~410 m.
- Hypothesis: the terrain-aided altimeter linearises a nonlinear terrain model, so it shrinks horizontal sigma along map gradients without real absolute info. (Half right, see E011.)
- Why it matters: a policy that reads covariance is only as good as the covariance.

### E010, 2026-10-14: Pure PPO vs residual PPO
- PPO from scratch, 40k steps (135 s): **0% safe landings**. Landing from a random policy is a hard exploration problem.
- Switched to residual RL: action = clip(baseline action + 0.5 x policy action), baseline action appended to the obs, log-std init -1.5. A zero residual reproduces the baseline exactly (tested).
- Reward: potential shaping toward the true target (privileged) + fuel/time costs + terminal bonus/penalty. Observation uses the estimate only.
- PPO: 8 envs, 512 steps x 8, batch 512, 8 epochs, lr 3e-4, gamma 0.995, GAE 0.95, MLP 2x256.
- Runs: `cov` vs `nocov` (covariance and fix-age features zeroed), seed 0, 600k steps, mixed_hard train sites.

### E011, 2026-10-14: Altimeter coupling test
Added `NavConfig.alt_mode`. `coupled` (default, used everywhere above) vs `decoupled` (height only updates the vertical state, height error from horizontal uncertainty `gx^2 Pxx + gy^2 Pyy` is added to the measurement noise, Schmidt-style). 100 episodes, seeds 10000+:

| Baseline guidance | Precise | Median / p90 err | Median NEES | Eps with median NEES > 18 |
|---|---|---|---|---|
| no_trn, coupled | 42% | 69 / 336 m | 69.4 | 71% |
| no_trn, decoupled | **9%** | 153 / 235 m | 15.3 | 38% |
| mixed_hard, coupled (first 100 of the 200 run) | 77.0% | n/r | 8.1* | 18.5%* |
| mixed_hard, decoupled | **84.0%** | 1.3 / 135 m | 7.2 | 1% |

(*from the 200-episode run)

- **No TRN:** the coupled altimeter carries real positioning info (terrain-contour matching). Turning it off drops precise 42% to 9%. So the overconfidence isn't just a linearisation artifact. Decoupled is much better calibrated but loses that info.
- **mixed_hard:** decoupling gives +7 pts and p90 168 to 135 m. Over the same 100 seeds, TRN gate rejections fell from 72 to 9 and fix acceptance rose 0.58 to 0.61, so an overconfident filter rejects good fixes. Fits the counts, not a controlled causal test. The later 200-episode decoupled run gave 83.0% vs 75.5% (table above).
- Consequence: strongest classical stack = baseline + decoupled altimeter. RL has to beat that, not just the original baseline. The E014 PPO runs used the coupled filter, so they're only valid within that environment.
- Not built: keep coupling but inflate its noise with a curvature/uncertainty term.

### E012, 2026-10-14: Where the misses come from (mixed_hard test)
Precise rate split by whether TRN gave at least one accepted fix:

| Policy | Zero-fix: precise (n) | At least one fix: precise (n) |
|---|---|---|
| baseline (coupled) | 7% (42) | 94% (158) |
| baseline (decoupled) | 6% (32) | 98% (168) |
| PPO with covariance | 10% (52) | 91% (148) |
| PPO without covariance | 17% (47) | 97% (153) |

- The baseline lands within a median 0.1 to 0.2 m (p90 0.3 to 1.0 m) of where it *thinks* the target is. So every miss is a navigation miss, not a guidance miss.
- By site and blackout (baseline, decoupled, n = 200): rough sites 0% zero-fix and 97 to 98% precise with or without blackout. Low-relief + blackout: 46% zero-fix, 56% precise. Low-relief, no blackout: 18% zero-fix, 80% precise. **All the headroom is on low-relief terrain.**
- TRN only works above 150 m AGL and the baseline drops below that at a median 70 s (20 episodes), so fixes basically have to happen in the first ~70 s. An early blackout can wipe them all out.
- So terminal-guidance learning can't help. Only changing *whether a fix is obtained* can: wait out a blackout (costs fuel and time) or detour toward textured terrain (untested then).

### E013, 2026-10-14: Covariance-triggered hold (hand-written)
`WaitGuidance`: while nav horizontal sigma > 40 m, AGL > 500 m, cumulative hold < `max_wait`, and propellant above a 220 kg reserve, hover instead of descending. Resume when a fix drops sigma. Uses only filter outputs.
- First attempt failed (113 of 200 timeouts). After hovering, the polynomial guidance computes time-to-go from a near-zero descent speed and plans a crawl. Fix: assume a nominal 15 m/s descent for time-to-go above 300 m AGL (hold variant only). Same stall mechanism as E005.1.
- `max_wait` picked on the **train** split (200 eps, decoupled), evaluated once on test:

| max_wait [s] | 0 | 30 | 50 | 70 | 80 | 90 |
|---|---|---|---|---|---|---|
| Precise (train) | 82.5% | 86.5% | 87.5% | **90.5%** | 89.0% | 81.0% |
| Timeouts (of 200) | 0 | 0 | 0 | 0 | 5 | 20 |
| p90 err [m] | 135 | 89 | 85 | 18 | 11 | 31 |
| Median fuel [kg] | 141 | 179 | 199 | 218 | 225 | 172 |

  The cliff at 80 s is the 250 s mission-time cap. 70 s selected.
- **Test (200 eps, never used for tuning):** decoupled 83.0% to **89.0%** precise, p90 110 to 60 m, zero-fix 32 to 21, fuel 152 to 220 kg (+68 kg of ~500 kg usable). Coupled 75.5% to **80.5%**, but p90 stays high (246 m).
- Disclosure: before the train sweep, one 90 s hold ran on the **test** split (`results/wait_mixed_hard_dec.json`: 79.0% precise, 26 timeouts). That's what prompted trying shorter holds. 70 s was chosen on train and the test number was computed once, but test wasn't perfectly untouched.
- Caveats: hand-designed rule, one parameter tuned on training sites, the 250 s cap directly shapes the optimum, and fuel isn't in the precise-landing metric.

### E014, 2026-10-14: First PPO results (negative)
600k steps each (~28 min), seed 0, mixed_hard train split, coupled filter. Training precise (rolling 500 eps) was ~0.77 at 30k and 0.79 (cov) / 0.81 (nocov) at 600k, basically flat. Baseline on the coupled *train* split wasn't measured, so can't say if that small rise is learning or noise. Held-out test, 200 eps, same coupled env:

| | Precise | p90 err | Fuel | Hard landings |
|---|---|---|---|---|
| baseline | 75.5% | 168 m | 152 kg | 6 |
| PPO, no covariance | 78.0% | 218 m | 147 kg | 1 |
| PPO, with covariance | 70.0% | 230 m | 119 kg | 8 |
| hold rule (E013) | 80.5% | 246 m | 220 kg | 2 |

- **H1 (covariance in obs helps): not supported.** cov is 8 pts below nocov. SE of the difference is ~4.4 so it's not decisive, but there's definitely no support.
- **H2 (RL beats fixed guidance on p90): not supported.** p90 is worse for both PPO runs (230 and 218 vs 168 m).
- cov used less fuel (119 kg) and had more hard landings (8): it traded safety margin for propellant, which the reward allows.
- The hold rule beats both learned policies, and the decoupled hold beats everything (89.0%).
- Why PPO missed the hold behaviour is untested. Candidates: (a) credit assignment over a 60 to 100 s delay with a sparse terminal signal, (b) residual authority 0.5 and small initial noise keep it near the baseline, (c) 600k steps is small, (d) the coupled filter's overconfident covariance is a bad input (tested in E018).
- One seed, one reward, one budget. "First attempt didn't work", not "RL can't help".

### E016, 2026-10-14: First real lunar terrain (PGDA LOLA 5 m DEMs)
**Data.** Two 8 x 8 km tiles from the PGDA polar DEMs, block-averaged 5 m to 10 m (`scripts/crop_dem.py`): Site 04 (Shackleton rim, train split) and Site 23 (Malapert massif, test split). Height std 314 m (Site 04) / 271 m (Site 23) vs ~18 m RMS synthetic. Roughly 14 to 35% of cells under 10 deg slope. 20 m onboard map vs 10 m truth: 0.33 / 0.28 m RMS. 164 / 150 candidate targets, clustered in a small part of each tile. About 90% of pixels are interpolated between laser tracks, so surfaces are smoother at small scales than real ground. Runs use `real+dec`, blackout prob 0.4.

**Results (100 episodes each, SE ~2 pts)**

| Policy, tile | Precise | Safe | Median err | Zero-fix eps | Median fuel |
|---|---|---|---|---|---|
| baseline, Site 23 (test) | 95% | 96% | 1.2 m | 0 | 149 kg |
| hold rule 70 s, Site 23 (test) | 95% | 96% | 1.3 m | 0 | 142 kg |
| baseline, Site 04 (train) | 98% | 98% | 1.2 m | 0 | 130 kg |

- TRN produced a fix in every episode, blackouts included. Baseline already lands 95 to 98% precisely and the hold rule changes nothing.
- Hypothesis (untested): relief ~15x the synthetic sites makes matching rarely ambiguous. Consistent with E012 but the tile wasn't varied in isolation.
- Doesn't show: (a) one tile per split with clustered targets, so two places, not "lunar terrain"; (b) interpolation may make matching easier; (c) steep rims/massifs are the opposite of the flat low-relief case the hold rule targets. A flat crater floor (like Haworth) is the informative next test.
- PPO not run on real tiles here.

### E018, 2026-10-14: PPO on the decoupled filter (tests E014 explanation d): still negative
Same as E014 (residual PPO, k = 0.5, 600k steps, 8 envs, seed 0) but `mixed_hard+dec`. Held-out test, 200 eps, deterministic policy:

| | Precise | Safe | p90 err | Fuel | Zero-fix eps |
|---|---|---|---|---|---|
| baseline | 83.0% | 99.0% | 110 m | 152 kg | 32 |
| PPO, no covariance | 82.5% | 98.5% | 104 m | 132 kg | 34 |
| PPO, with covariance | 59.0% | 74.5% | 161 m | 114 kg | 41 |
| hold rule 70 s (E013) | 89.0% | 99.5% | 60 m | 220 kg | 21 |

- PPO nocov matches the baseline (82.5 vs 83.0, SE ~2.7 each) with ~20 kg less fuel. PPO cov is much worse, 50 of 200 episodes ended in a hard landing.
- E014 explanation (d), overconfident coupled covariance, is not supported: with a calibrated filter cov got worse, not better.
- Training curves ended at 0.72 precise (cov, falling over the last 100k steps, so not stable) and 0.88 (nocov). The gap between nocov's training figure and its deterministic test figure wasn't investigated.
- Hypotheses (untested): reward lets the policy trade touchdown safety for propellant (cov policies used the least fuel and had the most hard landings in both E014 and E018); covariance features let the policy overfit training terrain.
- Where this leaves RL: two experiments, one seed each, neither beats the hold rule, neither learned to hover. Residual control around the baseline doesn't find the hold behaviour. Next RL experiment should change *what the policy decides* (high-level hold/detour), not train longer.

### E019, 2026-10-15: Random-window PPO on real PGDA terrain, training curve only
Run locally: `train_ppo_windows.py <Site TIFF> cov 0 600000 48` (48 random 8 km training windows, 600k steps, 574 s). Only the printed training log is available, no held-out eval.
- Rolling-500 precise (stochastic, training windows): 0.92 at 30k, 0.97 at 90k, ~0.94 through 300k, down to 0.81 by 420 to 450k, back to 0.90 at 510k, 0.78 at 570k, **0.65 at 600k**. The saved final model is the weakest point of the run.
- Baseline alone is at 95 to 98% on these tiles (E016), so the first 300k steps look like "policy ~ baseline, no headroom", and later updates made it worse.
- Hypotheses (untested): late PPO instability (policy-std or value drift on a nearly saturated reward); cov inputs overfit the training windows; the drop is a logging-window artefact.
- Tooling change: checkpoint saved every log step plus a best-by-training one (optimistic: stochastic policy, training windows).

### E020, 2026-10-15: Random-window PPO tested on a held-out real site: no gain, late training degrades
Run locally, printed summaries only (no JSONs). Train: 48 random 8 km windows of the Site 04 DEM (`--frac 1.0`), 600k steps, seed 0, `cov` and `nocov`, checkpoints every 30k. Test: 200 episodes over 24 random windows of the **Site 23** DEM (never trained on), decoupled filter, 40% blackouts, same windows and seeds for every policy. SE near 97% at n = 200 ~1.1 pts.

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

Training-curve endpoints (rolling 500 stochastic eps): cov 0.71, nocov 0.91.

- No PPO checkpoint beats the baseline. Baseline, hold and the good checkpoints are all within one SE. When the baseline lands, it's within 2.3 m (p90).
- Late training hurts: final cov is clearly worse (85.5%, 25 hard landings), final nocov mildly worse (95.0%). Only checking the final checkpoint would have misreported RL.
- Across cov checkpoints median fuel falls (156, 140, 137, 120 kg) while hard landings jump at the final one. PPO p90 error is 2 to 7x the baseline's. Fits the "safety for propellant" idea, not proven.
- The baseline's 2.5 pts of loss are 3 `unsafe_site` + 1 hard landing, not nav misses. Whether the 20 m onboard map can even see those hazards is untested.
- Hold rule gives nothing (zero-fix 1 of 200).
- Limits: one seed; one test site; PGDA is ~90% interpolated; the 24 windows overlap inside one DEM; best-by-training checkpoint picked on training windows only.
- Takeaway: on rugged real terrain there's no nav headroom and residual PPO adds nothing. An RL result needs real headroom: flat low-relief real terrain, or hazard retargeting inside the 50 m zone.

### E021, 2026-10-15: Haworth (PGDA, 29.8 x 29.8 km), proper geographic split: no headroom either
Run locally, JSON/CSV bundle (`results/haw_*.json`, `results/summary_windows.csv`) checked against the table. Relief std 704 m over the whole map (crater walls included), ~50% of the map under 10 deg slope in a coarse 33 m view (understates fine-scale roughness). Left 70% of columns = train, right 30% (8.9 km wide) = held-out test. 24 random windows, 200 eps, decoupled, 40% blackouts. SE ~0.5 pts at 99.5%, 1.4 at 96%, 2.5 at 85%.

| Run | Precise | Hard / unsafe / timeout | p90 err | Fuel |
|---|---|---|---|---|
| baseline, held-out strip | 99.5% | 0 / 0 / 0 | 2.1 m | 147 kg |
| hold rule 70 s, held-out strip | 99.5% | 0 / 0 / 0 | 2.2 m | 141 kg |
| baseline, train region | 96.0% | 0 / 7 / 0 | 2.1 m | 150 kg |
| hold rule, train region | 96.5% | 0 / 6 / 1 | 2.1 m | 142 kg |
| Site 04 PPO zero-shot, cov best / step 300k | 98.5% / 98.5% | 0/2/0 each | 11.5 / 10.5 m | 143 / 142 kg |
| Site 04 PPO zero-shot, nocov best / step 300k | 96.5% / 99.0% | 0/1/5 ; 0/0/1 | 6.0 / 6.6 m | 151 / 153 kg |
| Haworth PPO cov, best / 300k / final, held-out strip | 99.5% / 85.0% / 89.5% | 0/0/0 ; 28/0/0 ; 0/0/19 | 3.7 / 9.3 / 13.8 m | 143 / 119 / 147 kg |
| Haworth PPO nocov, best / 300k / final, held-out strip | 99.5% / 93.0% / 94.0% | 0/0/0 ; 0/1/13 ; 8/2/0 | 3.4 / 7.2 / 9.3 m | 155 / 156 / 118 kg |
| Haworth PPO cov, best / 300k / final, on Site 23 | 97.5% / 91.5% / 82.5% | 1/3/0 ; 12/3/0 ; 1/3/30 | 3.7 / 9.8 / 12.7 m | 144 / 126 / 158 kg |
| Haworth PPO nocov, best / 300k / final, on Site 23 | 97.5% / 94.5% / 89.5% | 1/3/0 ; 1/4/4 ; 16/3/0 | 3.5 / 6.6 / 9.3 m | 153 / 151 / 116 kg |

- Baseline is at 99.5% on the held-out strip with 1 zero-fix episode of 200, and the hold rule changes nothing. No policy beats the baseline in any row. Best-by-training checkpoints tie it, later ones are worse, finals are the worst (same as E019, E020). Failure modes differ per checkpoint: hard landings (cov 300k: 28 of 200) or timeouts (cov final: 19; nocov 300k: 13).
- Three real sites now (Site 04, Site 23, Haworth): baseline precise 95 to 99.5%, TRN almost always fixes. The low-relief, no-fix regime that gives 83% on synthetic mixed_hard just doesn't show up on real tiles.
- Hypothesis (untested): big-scale relief makes matching rarely ambiguous at 20 m map resolution, and the ~90% interpolation might also play a role.
- Limits: one seed per variant; held-out strip and train region are parts of one map (possible terrain-type shift, not inspected); windows overlap within a map.

### E022, 2026-10-15: Stressing navigation on real tiles
100 eps per cell, two 8 km tiles (one window each, site04 = train, site23 = test), decoupled filter, 20-cell search window. Settings: default; big_init (300 m initial error); long blackout (every episode, 60 to 100 s); both. SE up to ~4 pts.

| Tile | Setting | Baseline precise (safe) | Hold precise (safe) | Baseline non-success |
|---|---|---|---|---|
| site04 | default | 98% (98) | 98% (98) | 1 unsafe, 1 hard |
| site04 | big_init | 83% (90) | 83% (88) | 8 timeout, 2 unsafe |
| site04 | long blackout | 99% (99) | 99% (99) | 1 unsafe |
| site04 | both | 82% (89) | 82% (88) | 8 timeout, 3 unsafe |
| site23 | default | 95% (96) | 95% (96) | 4 unsafe |
| site23 | big_init | 78% (80) | 78% (80) | 3 timeout, 13 unsafe, 4 hard |
| site23 | long blackout | 94% (96) | 94% (96) | 4 unsafe |
| site23 | both | 77% (81) | 78% (82) | 5 timeout, 10 unsafe, 4 hard |

- Long blackouts do nothing on these tiles (TRN still fixes, zero-fix 0 to 1 of 100), so the hold rule has nothing to do.
- 300 m initial error costs ~15 to 18 pts on both tiles. When it lands, landing error stays small (p90 3 to 5 m), the losses are timeouts, unsafe and hard touchdowns.
- The "guidance failure after a late correction" reading of this was wrong, see E023. The table is still correct for this (narrow-window) config.

### E023, 2026-10-15: Correction to E022, the big_init losses were the search window
The default window is 20 cells = 400 m on the 20 m onboard map, which doesn't cover 3 sigma once init_pos_h >= 133 m. Test (`scripts/sweep_init.py`): baseline, 100 eps per cell, sigma 120 / 300 / 600 / 1000 m, window either 20 cells or widened to ceil(3 sigma / 20) + 2. Start range 1.0 to 1.6 km from target. SE up to 4 pts.

| Tile | sigma | Precise, narrow (20 cells) | Precise, wide (cells) | Wide: non-success outcomes; zero-fix eps |
|---|---|---|---|---|
| site04 | 120 | 98% | 98% (20) | 1 unsafe, 1 hard; 0 |
| site04 | 300 | 83% | 99% (47) | 1 unsafe; 0 |
| site04 | 600 | 24% | 99% (92) | 1 unsafe; 0 |
| site04 | 1000 | 12% | 99% (152) | 1 unsafe; 0 |
| site23 | 120 | 95% | 95% (20) | 4 unsafe; 0 |
| site23 | 300 | 78% | 92% (47) | 4 unsafe, 2 timeout, 1 hard; 0 |
| site23 | 600 | 23% | 73% (92) | 15 unsafe, 3 timeout, 3 hard; 17 |
| site23 | 1000 | 11% | 80% (152) | 8 hard, 6 unsafe, 4 timeout; 9 |

- The narrow window explains the E022 big_init losses on site04 (83% to 99%) and most on site23 (78% to 92%). With the narrow window, landing error is hundreds of metres at sigma >= 600 (p90 ~450 to 580 m): the matcher can't find the true position.
- Properly sized window: site04 holds 99% up to 1000 m. Site23 degrades at 600 to 1000 m (73 to 80%) with 9 to 17 zero-fix eps of 100 plus unsafe/hard touchdowns.
- Retraction: the E022 claims "guidance failures after a late correction" and "first real-terrain regime with RL headroom" are withdrawn for site04 and weakened for site23.
- Hypothesis (untested): on site23 at big initial error, a bigger search area makes ambiguous matches more likely, so what's left might be a navigation-ambiguity problem where hold/detour could help. Trajectories not inspected.
- Limits: one window per tile; sigma 600 to 1000 m with a 1.0 to 1.6 km start is extreme and wasn't checked against real mission numbers; wide cells take minutes to run.

### E024, 2026-10-15: Why does the baseline fail on site23 at large initial error?
`scripts/diagnose_failures.py data/site23.npy <sigma> 60 [tag] [policy]`: 60 eps (seeds 10000 to 10059), widened window, each non-precise episode classified from its history. SE ~6 pts. Files: `results/diag_site23_{600,1000}[_wait].json`.

| Setting | Precise | zero_fix (ambiguous, no fix) | false_fix (>100 m wrong fix accepted) | nav_ok_hazard (nav err <= 15 m, hard/unsafe touchdown) | other (mostly timeouts) |
|---|---|---|---|---|---|
| baseline, sigma 600 | 43 / 60 (72%) | 10 | 2 | 3 | 2 |
| baseline, sigma 1000 | 51 / 60 (85%) | 1 | 2 | 5 | 1 |
| hold rule 70 s, sigma 600 | 39 / 60 (65%) | 9 | 2 | 6 | 4 |
| hold rule 70 s, sigma 1000 | 42 / 60 (70%) | 2 | 1 | 7 | 7 (+1 nav_bad) |

- **Type A, ambiguity / no fix:** biggest bucket at sigma 600 (10 of 17 failures). All 10 were `rejected_ambiguous` the whole way, touching down 470 to 1220 m off target, 9 of 10 on unsafe ground.
- **Type B, nav fine but bad touchdown:** 3 to 5 eps per 60 (nav error <= 15 m, hard landing or unsafe site), plus 1 to 2 timeouts with good nav. (The "pure guidance problem" reading was withdrawn in E026.)
- **Type C, false fixes:** 1 to 2 per 60, ending 900 to 1900 m off.
- The hold rule doesn't help: 65% vs 72% at sigma 600, 70% vs 85% at sigma 1000, with more timeouts and hard landings. Ambiguity bucket unchanged (9 vs 10).
- Count note: E023 counts zero-fix over all eps (9 at sigma 1000), here only non-precise eps (1). The difference is lucky precise landings without any fix, not a conflict.
- Hypothesis (untested): hovering keeps the same viewpoint, so it can't resolve an ambiguous match. A change in altitude or lateral position might.
- Limits: one tile, 60 eps per cell, classification thresholds (100 m, 15 m, 300 m) are arbitrary picks.

### E025, 2026-10-15: What happens in the "nav fine, touchdown bad" episodes?
`scripts/inspect_typeB.py` re-ran 11 type B / timeout episodes from E024 (site23, baseline; seeds 10035, 10037, 10038, 10055, 10056 at sigma 600; 10004, 10005, 10035, 10037, 10049, 10056 at sigma 1000). Small sample, case studies not rates.

| Pattern | Seeds | Evidence |
|---|---|---|
| Designated target is itself unsafe in truth | 10035 (both) | landed within 1 m of target, slope 10.1 deg vs 10 deg limit; targets are picked safe on the 20 m onboard map only, truth is the 10 m map |
| Reaches target, hits at 4+ m/s | 10037 (both), 10049 | error 6 to 9 m, vz 4.0 to 4.5 m/s (limit 2), 360 kg fuel left |
| Crashes short of target at speed | 10056 (both), 10004 | t = 55 to 62 s, 150 to 250 m from target, vh 18 to 22 m/s, vz 6 m/s, 423 kg fuel left, first fix at t = 19 s (10056) |
| Never lands, loiters to time limit | 10038, 10055 | timeout at t = 250 s, 27 to 57 m up, vz 0.5, 230 kg fuel left |

- Nobody ran out of fuel (230 to 420 kg left). End position error was 1 to 15 m in all but 10004, 10005, 10049.
- The baseline only enters terminal descent if altitude < 40 m **and** estimated horizontal offset < 25 m (`PolyGuidance.act`). Otherwise it keeps flying the polynomial gate law (gate 30 m above target, vz -3 m/s) with a time-to-go that collapses near the ground.
- First read: guidance deficiency from arriving over the gate with offset > 25 m after a late correction. **Partly withdrawn in E026** (altitude lock-out was at least part of it).
- 1 of 11 is an unsafe designated target (map-vs-truth mismatch), <= ~2% of episodes, hits every policy equally.

### E026, 2026-10-15: Nav filter flaw, altitude lock-out after big terrain fixes
Tracing why the new landing sequence (`LandingGuidance`, "baseline v2") still timed out at sigma 600: in seed 20014 the lander estimated 10.6 m above the target while 92 m up, with horizontal error < 1 m. Time series (baseline guidance): altitude error grows to -82 m while the horizontal estimate is still 520 m wrong (the altimeter fits the map under the wrong position). The first terrain fix then jumps the horizontal estimate and the -82 m altitude error stays for the rest of the episode.
- **Cause (code + trace):** the altimeter update is chi-square gated (1 dof, 10.8) with a small altitude variance. After a horizontal jump the map height under the lander changes by up to hundreds of metres, the residual blows the gate, and every later altimeter update is rejected. A flaw in the nav filter, not in the terrain or guidance.
- **Fix, `nav.alt_sync` (default off so old results reproduce, `--alt_sync` flag in window scripts):** after an accepted TRN fix, add (map height change)^2 to the altitude variance so the gated update can pull altitude back. A first version that shifted the altitude *state* by the map-height change was worse (22/40 vs 33/40 at sigma 600, altitude errors up to 355 m) and was dropped.
- **Results, site23, baseline guidance, widened search, paired seeds 10000+:**

| Setting | Precise without / with alt_sync | nav_ok_hazard | zero_fix |
|---|---|---|---|
| sigma 120 (n=100) | 95% / 95% | 4 / 4 | 0 / 0 |
| sigma 600 (n=60) | 72% / 78% | 3 / 1 | 10 / 10 |
| sigma 1000 (n=60) | 85% / 92% | 5 / 1 | 1 / 1 |

  Separate tuning seeds 20000+ (n=40): sigma 600 33/40 to 35/40, sigma 1000 34/40 with sync (no matched baseline). Remaining failures there are zero_fix, false_fix and one timeout.
- Good-nav bad-touchdown failures nearly vanish (3 to 1, 5 to 1 per 60). Ambiguity failures at sigma 600 unchanged (10 / 10). Nothing changes at sigma 120. The +6 / +7 precise gains are ~1 SE each (SE ~6 at n = 60), the type B drop is the clearer signal.
- **Corrections:** E025's "deficiency of the hand-written guidance" and E024's "type B is a pure guidance problem" are withdrawn pending a re-run of guidance variants with `--alt_sync`.
- Baseline v2 (`LandingGuidance`): first version approached too slowly (bug in the gating condition, fixed). Without alt_sync it ties the old baseline (33/40 vs 33/40 at sigma 600, tuning seeds, many identical failures), as expected if the cause was the filter. Not run with alt_sync yet.
- Hypothesis (untested): with alt_sync, the remaining failures at sigma 600 are mainly ambiguity and wrong accepted fixes (i.e. navigation).
- Limits: one tile, 40 to 100 eps per cell, the fix is a heuristic covariance inflation (not a re-derived filter), effect on Site 04, Haworth and the PPO runs not measured.

### E027, 2026-10-16: Perfect-navigation reference
`scripts/diagnose_failures.py <tile> 600 100 oracle_<tile> baseline --oracle` (`oracle_nav`: guidance sees the true state, TRN off for speed). Same baseline, same 100 eps (seeds 10000+).

| Tile | Precise with perfect nav | Failures |
|---|---|---|
| site23 | 96 / 100 | 4 unsafe_site, landing error 0 m |
| site04 | 99 / 100 | 1 unsafe_site, landing error 0 m |

- With perfect nav the baseline lands on target within 1 m every time. All 5 failures (of 200) are unsafe touchdowns exactly on the designated target: picked safe on the 20 m onboard map, unsafe on the 10 m truth (same artefact as seed 10035 in E025). So any policy that only controls *how it flies to the given target* tops out at 96% (site23) / 99% (site04). Going beyond needs picking a different, truly safe site (hazard detection + retargeting).
- Paired seeds 10000 to 10059, site23: fixed baseline (`--alt_sync`) fails 13 at sigma 600, the oracle also fails 1 of those (target artefact), so 12 of 13 are navigation. At sigma 1000: 5 failures, 4 navigation. At sigma 120 the fixed baseline (95%) is already at the reference (96%).
- Reading: at default error there's no headroom for any flying policy. At sigma 600 the 78% to 96% gap is an information gap (no fix / wrong fix), closable only by getting more or better info (e.g. choosing where to fly so matching works). Separately there's a 1 to 4% gap from the unsafe-target artefact, closable only by retargeting.
- Limits: one tile each; the oracle removes all nav error including vertical; the reference depends on this env's target selection and 10 m truth safety rule.

### E028, 2026-10-16: Active perception, does flying toward better-textured ground fix the no-fix failures?
`DetourGuidance` in `trnrl/guidance.py`, run via `scripts/diagnose_failures.py data/site23.npy 600 60 detour_600 detour --alt_sync`, 60 eps (seeds 10000 to 10059), paired with the fixed baseline `s23_600_sync`. Hand-written rule: if horizontal 1-sigma > 60 m, no fix for 12 s and altitude > 400 m, do one lateral detour (up to 600 m, 15 m/s) toward the highest-relief cell on the onboard map (std of height over 300 m, `Site.map_tex`), then resume normal descent. One detour per episode.

| | Precise | zero_fix | false_fix | nav_ok_hazard |
|---|---|---|---|---|
| fixed baseline (alt_sync) | 47 / 60 | 10 | 2 | 1 |
| + hand-written detour | 47 / 60 | 10 | 2 | 1 |

- Same total, different episodes. The detour turned 3 of the 10 no-fix eps (10010, 10017, 10022) into precise landings (20 to 26 fixes each, 1 to 5 m error) and turned 3 precise eps into no-fix failures (10007, 10041, 10055). In those 3 the baseline's first fix had come late (25 to 27 s) and the detour moved the lander off the ground track that would have produced it. 53 of 60 eps have the same outcome and error within 5 m.
- Reading: a different ground track does produce fixes in a minority of ambiguous cases, so active perception can add information. But the relief-std texture proxy doesn't predict where matching works, and the rule can't tell which episodes need a detour. Picking detour/no-detour per episode with hindsight would give 50 / 60 (+5 pts), within noise and an upper bound for this rule only, not the idea.
- Limits: one tile, 60 eps, one rule, one detour per episode; seeds not split into tune/report sets (no tuning was done on this run).

### E029, 2026-10-16: Why the hand detour failed, bigger matching patch, learned hold/detour decision
Code: `trnrl/matchability.py`, `trnrl/detour.py` (`DetourGuidance2`, `DetourEnv`), `scripts/train_detour.py`, `scripts/eval_detour.py`, `--patch_n` flag.

**1. First fix is the bottleneck** (site23, sigma 600, alt_sync, 40 eps, 1231 fix attempts). Attempts with pre-attempt horizontal sigma under 60 m succeeded 863 / 863. Attempts above 450 m succeeded 8% of the time (368 attempts, all the rest `rejected_ambiguous`). Nothing in between. Among the high-sigma attempts, the new matchability score (`matchability_map`: second-peak separation outside the main lobe x lobe sharpness, onboard map only) predicts success with AUC 0.66. The relief-std score from E028 has AUC 0.33, i.e. points the wrong way. (An earlier version using post-update sigma leaked the label and isn't reported.)

**2. Matching patch size** (site23, sigma 600, alt_sync, 60 paired eps, baseline guidance):

| Patch (cells) | Precise | zero_fix | false_fix | nav_ok_hazard |
|---|---|---|---|---|
| 15 (default) | 47 / 60 (78%) | 10 | 2 | 1 |
| 23 | 53 / 60 (88%) | 6 | 0 | 1 |
| 31 | 54 / 60 (90%) | 5 | 0 | 1 |
| perfect navigation (E027, same seeds) | 59 / 60 | | | |

A bigger matching footprint is a classical fix that recovers most of the nav gap at big initial error. It assumes a sensor covering 460 to 620 m of ground at 20 m spacing from high altitude (not checked against a real instrument). With patch 31, remaining no-fix eps are ~8% on site23 and ~1% on site04.

**3. Training feasibility on real tiles.** With patch 31 at sigma 600, the learned-detour env found no episode needing a decision on site04 in 400 consecutive resets (basically nothing to learn from). On site23 roughly 1 in 12 (estimated).

**4. Learned hold/detour on synthetic `mixed_hard+dec`** (held-out test sites, 200 eps, seeds 10000+, same seeds for every policy, SE ~2.5 pts, alt_sync off as in E013):

| Policy | Precise | No-fix eps |
|---|---|---|
| baseline | 83.0% | 32 |
| hand-written hold (70 s) | 89.0% | 21 |
| always hover 25 s per decision (up to 3) | 87.5% | 22 |
| hand-written detour toward best matchability | 75.5% (43 timeouts) | 9 |
| PPO from scratch (1500 decisions, ~600 eps) | 81.5% | 35 |
| PPO warm-started from always-hover, seed 0 (6000 decisions, 2200 eps, 41 min) | 87.5% | 22 |
| same, seed 1 | 87.5% | 22 |
| perfect navigation (200 eps) | 99.5% | |

- Both warm-started runs end with exactly the always-hover outcomes. PPO found nothing better than hovering with this action set (hover, or lateral detour to 16 candidate points, 31 features including the matchability map). Training reward curves flat at 0.78 to 0.86.
- Reading (hypotheses): (a) the 10-pt gap between hold (87.5 to 89.0%) and perfect nav (99.5%) is episodes where no fix ever arrives, hovering doesn't create one and the lateral detours didn't either (they cost time). (b) On real tiles the lever turned out to be matching footprint and filter, not flight policy.
- Limits: 2 PPO seeds; one policy class and reward; matchability validated on one tile; real-tile numbers use alt_sync, synthetic ones don't; one sensor-footprint model.
- Doesn't show that no RL policy can beat these baselines, only that this action set, reward and training budget didn't. (The "learned = hover" reading is revised in E033.)

### E031, 2026-10-17: Realism layer (map registration error, correlated map noise, FOV-based patch)
Why: our median landing miss was 1.3 to 1.7 m vs 5 to 9 m for flown systems (Mars 2020 ~5 m, post-landing estimate 8.5 m or better; LVS map-relative < 40 m). Hypothesis: the idealised onboard map (derived from truth, no registration error).
- New knobs, all default OFF so E001 to E029 still reproduce:
  - `nav.reg_offset_m`: fixed per-episode map-to-world offset (1 sigma/axis), biases every TRN fix.
  - `env.map_noise_m` / `map_noise_corr_m`: correlated vertical error in the onboard map (also affects target selection).
  - `nav.fov_deg`: patch side = 2 * agl * tan(fov/2) in map cells, clipped 7..41, replaces fixed `patch_n`.
  - Flags `--fov --reg_offset --map_noise --map_corr`; preset `--realistic` = fov 40 deg, offset 6 m, noise 1 m (assumed values, not published, swept in part R).
  - NCC search is chunked now (wide search + big patch was getting OOM-killed).
- Smoke only: site23 sigma 120 with `--realistic`, median miss ~6 to 13 m over 12 eps (was ~1 m), now in the flown 5 to 9 m range. mixed_hard+dec N=6: 0.4 to 9.6 m vs 0.9 to 2.5 m legacy (excluding one 250 to 360 m failure in both).
- Suite reworked: parts A/B/C run under `--realistic` (fov60 replaces patch31 as the "larger sensor" variant), new part R (one-knob sensitivity + RL robustness under realism), part L (legacy conditions for comparison).

### E032, 2026-10-17: Final suite with the realism layer (parts A, B, D, E, L, R)
Source: `results/paper_tables.md` from `scripts/run_paper.sh`. n=200 for A, D, E, R-dec; n=100 for B, L and the R sweeps. Realism preset = fov 40 deg, 6 m offset/axis, 1 m map noise unless stated.
- **A (default 120 m, n=200):** baseline precise 96.5 / 99.0 / 98.5 / 95.5% (Site04 / Site23 / Haworth test / Haworth train). Perfect nav 98.5 / 97.5 / 98.0 / 97.5%. Median miss 7.4 to 8.4 m, p90 13 to 15 m. alt_sync identical to non-fixed at default error. Hold rule within noise of baseline.
- **B (n=100):** narrow 20-cell window 81% (sigma 300), then 24 / 25% (600) and 12% (1000) on both tiles. Wide window: Site04 97 / 99 / 99%, Site23 99 / 89 / 94%. +alt_sync: Site23 99 / 90 / 96%. +fov60: Site23 99 / 92 / 97%. +hold: Site23 99 / 91 / 96%. At Site23 sigma 600 nearly every failure is a zero-fix episode (7 to 9 of 7 to 10).
- **R sensitivity (Site23, alt_sync):** sigma 600 precise by fov 20 / 30 / 40 / 60 deg = 81 / 86 / 90 / 92%, zero-fix 17 / 12 / 9 / 7. Sigma 120 flat at 97 to 99%. Offset 0 / 3 / 6 / 12 m: median miss 3.8 / 5.5 / 8.4 / 14.8 m, precise unchanged (98 to 100% at 120, 89 to 91% at 600). Map noise 0 to 3 m: median miss 7.6 to 13.8 m, precise 96 to 99% (120) and 88 to 90% (600).
- **D (idealised map, n=200):** PPO nocov seeds 0 / 1 / 2 = 82.5 / 77.0 / 79.0%, cov = 59.0 / 66.5 / 79.5%, vs baseline 83.0%, hold 89.0%. Haworth PPO: s1 best 98.5 / final 87.5% (21 hard landings), s2 best 99.5 / final 98.0%. No run beats baseline beyond SE. The earlier "cov worse than nocov in every run" claim is **false** with 3 seeds (seed 2: cov 79.5 vs nocov 79.0).
- **E / R-dec (mixed_hard+dec, n=200):** baseline 83.0 nominal, 73.0 realism (zero-fix 32 to 49). Hold 89.0 / 78.0. Hand detour 75.5 / 76.5 (43 and 31 timeouts). Always-hover 87.5. Learned seeds 0, 1, 2 identical to always-hover (87.5 nominal, 77.5 realism, identical p90). Perfect nav 99.5.
- **L (n=100, idealised map):** Site23 sigma 120 legacy 95 / sync31 96%. Sigma 600 legacy 73 / sync31 88%.
- The three learned-decision "seeds" are really one policy: measured fact is identical outcomes. Cause (hover-logit bias 4.0, lr 1e-4, 6000 decisions, greedy action never flips) was read from the code, not tested with a bias sweep. Don't report as 3 independent results.
- Hypotheses (untested): baseline above the perfect-nav number on Site23 (99.0 vs 97.5, within SE) because the 6 m offset moves some touchdowns off an unsafe designated cell (checked in E038). alt_sync matters less under the fov-40 patch (89 vs 90, 94 vs 96) because earlier first fixes reduce lock-out.
- Site23 sigma 1000 (94 to 97%) above sigma 600 (89 to 92%) was read as noise here (SE 2 to 3). E038 shows it isn't.
- Hold rule +5 to +6 pts on mixed_hard (both conditions) is ~1.5 to 2 SE, suggestive only (paired CI in E037).

### E033, 2026-10-17: Decision-layer headroom and a one-line rule that captures it (supersedes the E029 reading of the learned detour)
`scripts/decision_headroom.py` tries each first-decision action (0 none, 1..8 detour 400 m, 9..16 detour 800 m, 17 hover), then no further detours, on the same episodes. `scripts/fit_decision_policy.py` fits a chooser on training-site logs and evaluates exactly on test logs (every action's outcome known per episode). mixed_hard+dec, test n=200 (110 reach a decision point), train n=400 (217 with a decision point), seeds 10000+i (test) / 50000+i (train), separate sites.
- **Headroom, nominal test:** none 81.5%, hover 86.0%, mean over 16 single detours 89.8% (range 86.0 to 92.5%), hindsight best of all 98.5%. Realism layer: none 73.5, hover 76.0, mean detour 79.3 (75.5 to 83.5), hindsight 99.5.
- Hand matchability rule at the FIRST decision only: 90.0% (zero-fix 17) vs 75.5% with up to 3 decisions (43 timeouts). So the E028 failure was repeated decisions, not detouring itself.
- Matchability rule 90.0% vs random detour 89.7%: no direction info in the matchability map as used.
- Detour options binned by alignment with the direction to the target (held-out): away (-1..-0.5) 89.1% precise (n=586), toward (0.5..1) 75.9% (n=586).
- **One-line rule "detour 800 m directly away from the target": 93.5% test, 92.7% train** (nothing fitted). At 400 m: 91.0% test.
- Conditional-logit chooser (shared weights, 18 intercepts, C=0.1 by 5-fold CV on train): 94.5% test. Paired bootstrap vs none +13.0 (CI +8.0..+18.0), vs hover +8.5 (+5.0..+12.5), vs best fixed action +5.0 (+1.5..+8.5), vs matchability rule +4.5 (+1.0..+8.5), vs random detour +4.8 (+2.4..+7.3). 30 bootstrap refits: 92.5 to 95.5%, mean 93.7%. Without alignment features 89.5% (= best fixed action). Without matchability 94.0%.
- Learned chooser minus the one-line rule: +1.0 pt at n=200, not significant (2 episodes).
- Hypothesis (untested): away helps because of more time plus a second viewing geometry for the first fix. No experiment separated the two (E040 starts on it).
- Hindsight 98.5% is optimistic (best of 19 tries per episode, noise stream shared across actions).
- E029's "learned detour equals hover" reflects the warm start (bias 4.0 on hover logit, lr 1e-4, 6000 decisions), not that detouring can't help.
- Limits: synthetic low-relief benchmark, nominal conditions for the fit; supervised learning on full-information logs, not RL; no real tiles.

### E034, 2026-10-17: Part C and decision headroom under realism
- **C (Site 23, 60 eps each, realism layer):** sigma 600: legacy-window 53 precise (zero_fix 5, nav_ok_hazard 2), +alt_sync 54 (zero_fix 5, nav_ok_hazard 1), +fov60 54 (zero_fix 5, nav_ok_hazard 1). Sigma 1000: 57 (nav_ok_hazard 1, nav_bad 1, other 1), 59 (nav_ok_hazard 1), 59 (nav_ok_hazard 1). No zero-fix eps at 1000 m. Matches B (89 / 90 / 92% at 600 m).
- **H realism headroom, train n=400 (165 with a decision point):** none 73.0, hover 70.8, mean single detour 77.2, best single 80.5, hindsight 95.8. **Test n=200:** none 73.5, hover 76.0, mean detour 79.3, best 83.5, hindsight 99.5. **Nominal train n=400:** none 84.5, hover 86.0, mean detour 89.8, best 93.5, hindsight 99.5.
- Hover vs none flips sign across splits (+2.5 test, -2.2 train, SE ~2 to 3), not reliable. Mean single detour beats none by +5.8 (test) and +4.2 (train).
- The realism-layer chooser fit was missing from this run (fit step failed or didn't run), done in E035.

### E035, 2026-10-17: Decision chooser fitted under the realism layer
`fit_decision_policy.py` on P_H_train_realistic (n=400, 165 with a decision point) and P_H_test_realistic (n=200, 88 with a decision point). Needed scikit-learn (was missing, installed).
- **Test (precise = within 50 m, n=200):** none 73.5%, hover 76.0%, random detour 79.3%, best fixed action (chosen on train) 81.0%, matchability rule 81.5%, conditional logit 83.5% (C=1.0), away-800 rule 84.5%, away-400 rule 81.5%, hindsight best 99.5%.
- Train 5-fold CV score of the chooser on decision episodes only: 0.56 to 0.58 (nominal: 0.87 to 0.89).
- Chooser vs none +10.0, vs best fixed +2.5, vs random detour +4.2 (point estimates, bootstrap in E036). Chooser is 1.0 below the one-line away-800 rule.
- Reading (untested): under realism the chooser has no demonstrated edge over the one-line rule. Decision-time features predict little of the per-episode variation (CV 0.56 to 0.58 vs hindsight 99.5%), possibly because outcomes are dominated by noise draws not visible at decision time.
- Detouring away from the target helps in both settings (nominal 93.5 vs 81.5 none; realism 84.5 vs 73.5 none). That's the robust finding. The learned layer adds nothing measurable beyond it.
- Scope: only 88 test eps reach a decision point, so a 1 to 3 pt difference is 2 to 6 episodes. Synthetic benchmark, realism knob values assumed.

### E036, 2026-10-17: Paired bootstrap for the realism-layer chooser
`scripts/bootstrap_decision.py`: 2000 paired resamples of the 200 test eps with the chooser fixed (C=1.0), plus 100 refits on resampled train decision eps. Output `results/P_H_boot_realistic.json`.
- Points, 95% CI: chooser minus none +10.0 (4.5, 15.5); minus hover +7.5 (2.0, 13.0); minus random detour +4.2 (0.4, 8.1); minus best fixed +2.5 (-2.5, 7.5); minus matchability rule +2.0 (-3.5, 7.5); minus away-800 rule -1.0 (-6.0, 4.0).
- Refit range over train resamples: 79.5 to 87.5%, mean 84.0%.
- Reading: chooser beats no detour, hover and random detour, but can't be told apart from the best fixed action or the rules at this sample size. (More decision episodes would separate them: untested.) Nominal-case bootstrap was in E033.

### E037, 2026-10-17: Paired CIs for the main tables, Haworth checkpoint selection check
`scripts/paired_ci.py` pairs episodes by seed across result JSONs (precise = success and error <= 50 m, verified equal to the summaries for every file), 5000-resample paired bootstrap + exact sign test. Output `results/paired_ci.md`, also appended to `paper_tables.md`. 70+ comparisons over A, B, D, E, R. 95% paired CIs, points:
- **A:** baseline minus perfect nav: Site 04 -2.0 (-4.5, 0.0), Site 23 +1.5 (-1.0, +4.0), Haworth test +0.5 (-1.0, +2.5). alt_sync window fix vs legacy: 0 discordant eps on all tiles at default settings.
- **B:** wide vs narrow window +64 to +87 pts at 600 / 1000 m (CIs nowhere near 0). alt_sync vs wide: Site 23 600 m +1.0 (0.0, +3.0), 1000 m +2.0 (0.0, +5.0), Site 04 0.0. sync+fov60 vs sync, Site 23 600 m +2.0 (0.0, +5.0). Hold vs baseline within +/-1 pt everywhere.
- **R (Site 23, 600 m, n=100):** fov60 minus fov20 +11.0 (+5.0, +17.0), sign p=0.001. fov20 vs preset -9.0 (-15.0, -4.0). fov30 vs preset -4.0 (-8.0, -1.0), p=0.125. fov60 vs preset +2.0 (0.0, +5.0). Offset 0 / 3 / 12 and map noise 0 / 2 / 3 vs preset: all intervals include 0.
- **E / D synthetic (n=200):** hold minus baseline +6.0 (+2.0, +10.0), p=0.008. Under realism +5.0 (+0.5, +9.5), p=0.064. Hand detour vs baseline -7.5 (-14.0, -1.5) nominal, +3.5 (-2.5, +9.5) realism. Warm-started learned seeds = hover (identical). Learned minus baseline +4.5 (+1.0, +8.5) = hover's gain. Learned minus hold -1.5 (-4.0, +1.0).
- **PPO residual vs baseline:** nocov s0 / s1 / s2: -0.5 / -6.0 (-11.0, -1.5) / -4.0 (-8.5, +0.5). cov: -24.0 (-31.5, -16.5) / -16.5 (-23.0, -10.0) / -3.5 (-8.0, +0.5). Haworth best: s1 0.0 (-2.0, +2.0), s2 +1.0 (0.0, +2.5). Haworth final: s1 -11.0 (-15.5, -6.5), s2 -0.5 (-2.5, +1.5).
- Checkpoint selection check (`train_ppo_windows.py`): `_best` = highest rolling precision logged during training (stochastic policy, training windows only). Test strip is never used for selection. Optimistic and noisy, but not leaked.
- Caveats: paired CIs treat the 200 (or 100) seeds as the sample, training-seed variance is separate (3 seeds, learned-decision seeds identical). P_E baseline (83.0%) differs from the H "none" rate (81.5%) because H logs are a separate run with different random streams. Both reported as is.

### E038, 2026-10-17: Site 23 extended runs (400 eps, part X) and baseline vs perfect-nav diagnosis
`PART=X NX=400` (10 jobs, ~5.5 h on a Ryzen 6600H). The sigma1000 sync_hold job was killed after 137 CPU-min so that cell is missing. First 100 seeds equal part B, so results pair with B.
- **Precise (<= 50 m, n=400):** sigma 600: wide 86.0, sync 88.0, sync+fov60 89.2, fov30 85.5, fov20 76.8, hold 88.0. Sigma 1000: wide 90.0, sync 92.2, sync+fov60 95.5.
- **alt_sync vs wide:** +2.0 (+0.8, +3.5) at 600 m, +2.2 (+1.0, +3.8) at 1000 m (8/0 and 9/0 discordant).
- **fov at 600 m:** 20 vs 40 -11.2 (-14.8, -8.0). 30 vs 40 -2.5 (-4.8, -0.2), p=0.052. 60 vs 40 +1.2 (-0.2, +2.8), p=0.18. 60 vs 20 +12.5 (+9.0, +16.2). Fov 60 vs 40 at 1000 m: +3.2 (+1.5, +5.2), p=0.001.
- **hold vs baseline at 600 m:** +0.0 (-1.8, +1.8).
- **sigma 1000 minus sigma 600 (sync):** +4.2 (+1.5, +7.2), p=0.006. The 1000 m cell really is easier, not noise (fixes the E032 reading). Zero-fix eps: 38 (600 m) vs 16 (1000 m). Failures at 600 m are mostly unsafe_site (39 of 48) together with zero fixes. Why it's easier is untested (guesses: interaction of initial offset, the 30 m patch footprint and relief along the approach).
- **Baseline vs perfect nav on Site 23 (n=200, 7 discordant eps):** in 5, the oracle lands within 0.1 m of the designated target and that target is unsafe on the truth map (the baseline misses by 7 to 14 m and lands on a safe cell). In 2, the baseline fails and the oracle succeeds. That's the measured reason the baseline can sit above the reference, so "ceiling" got renamed "perfect-navigation reference".

### E040, 2026-10-17: Mechanism check 1, time vs direction (nominal mixed_hard+dec)
`scripts/mechanism.py` (mode `geom`): time-matched hover 25 / 60 s vs detours away / toward / perpendicular at 400 / 800 m. 110 of 200 test eps have a decision point.
- **Precise on decision eps:** none 68.2; hover 25 s 76.4; hover 60 s 79.1; away 400 / 800 m 85.5 / 90.0; perpendicular 400 / 800 m 82.7 / 85.5; toward 400 / 800 m 68.2 / 75.5.
- **Accepted fix after the decision:** none 68.2% (= its precise rate), hover25 78.2, hover60 80.0, toward800 77.3, perp800 89.1, away800 92.7. Median total flight time: hover60 195 s, away800 201 s, perp800 184 s. Mean accepted fixes after decision: none 7.1, hover60 9.5, away800 10.9.
- **Paired CIs (n=110, 95% bootstrap, points):** away800 - hover60 +10.9 (+5.5, +17.3), 13/1, p=0.002. away800 - toward800 +14.5 (+6.4, +22.7). away400 - toward400 +17.3 (+9.1, +26.4). hover60 - none +10.9 (+2.7, +19.1). hover25 - none +8.2 (+0.9, +15.5). away800 - perp800 +4.5 (-3.6, +12.7). toward800 - hover60 -3.6 (-10.0, +2.7).
- Reading: time alone gives some gain (+8 to +11 pts over none), but at matched flight time away beats hover 60 s by ~11 pts and toward by ~14.5. Direction matters. Away vs perpendicular can't be separated. Why direction matters is untested (candidate: longer return leg gives more matching attempts at low altitude, not terrain matchability, which carries no direction info).
- Still to run (modes `noise` and `fix` of `mechanism.py`, smoke-tested only): replay each decision episode with different measurement noise to see how much of the hindsight cap is luck; first-fix time and altitude-state error at touchdown for legacy / fov40 / fov60 x alt_sync off/on (hypothesis: alt_sync matters less under realism because the first fix comes earlier).
