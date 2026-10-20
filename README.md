# lunar-trnrl

An open 3-DoF lunar powered-descent benchmark on real polar terrain (LOLA digital elevation
models), with a full terrain-relative navigation (TRN) filter in the loop. It is built to ask one
question: **where can a learned guidance policy add anything over a classical baseline?**

Companion paper: *Where Does Learned Guidance Help? A Real-Terrain Lunar Landing Benchmark*
(Nimesh Chauhan) Archived code: https://doi.org/10.5281/zenodo.23497770

## What is in the box

| Path | Contents |
|---|---|
| `trnrl/` | The library: terrain handling (`terrain.py`, `dem_source.py`), matchability map, 9-state EKF with IMU, Doppler, terrain-aided altimeter and TRN fixes (`nav.py`), Gymnasium descent environment (`env.py`), baseline polynomial guidance (`guidance.py`), residual-PPO wrapper (`residual.py`), decision layer (`detour.py`), benchmark definitions (`benchmarks.py`), window sampling (`windows.py`) |
| `scripts/` | Training, evaluation, sweeps, mechanism checks, paired statistics, table and figure generation |
| `tests/` | `pytest` suite (20 tests) |
| `results/` | Result JSON files behind every table in the paper, plus `paper_tables.md` and `paired_ci.md` |
| `figures/paper/` | Paper figures (PDF and PNG) |
| `sandbox/` | A small browser 3D front end for flying the simulator (see `SANDBOX.md`) |
| `RESEARCH_LOG.md` | Chronological research log, including failed experiments and a bug found along the way |

## Install

```bash
conda create -n trnrl python=3.11
conda activate trnrl
pip install -r requirements.txt
python -m pytest tests -q
```

Tested with Python 3.11.0, gymnasium 1.4.0, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1,
stable-baselines3 2.9.0 and torch 2.14.1. `requirements.txt` lists the dependencies; for exact
reproduction, install those versions.

## Get the terrain data

The DEMs are public downloads and are not stored in the repository. See `data/README.md` for
sources and the expected file names. Then build the crops used by the synthetic benchmarks:

```bash
python scripts/crop_dem.py data/Site04_final_adj_5mpp_surf.tif --size_km 8 --out data/site04.npy
python scripts/crop_dem.py data/Site23_final_adj_5mpp_surf.tif --size_km 8 --out data/site23.npy
```

Other options: `--centre` (crop around the tile centre), `--out_dx` (output cell size in metres,
block-averaged, default 10) and `--dx` (metres per pixel if not stored in the file).
TODO: confirm the Site23 command and whether either crop used `--centre`.

## Reproduce the paper

Each experiment writes `results/<tag>.json` and logs to `logs_paper/<tag>.log`
(tags start with `P_`).

```bash
# smoke test of every command (a few minutes)
QUICK=1 bash scripts/run_paper.sh

# full run, in parts: A default regime + ceiling, B initial-error sweep, C failure diagnosis,
# D residual PPO seeds, E learned hold/detour, R realism sweeps, L legacy-patch comparison,
# H decision headroom, X extra n=400 reruns, S/S2 decision-PPO and rule runs, T collect tables
PART=all N=200 NB=100 JOBS=2 HAW=/path/to/Haworth_final_adj_5mpp_surf.tif \
  bash scripts/run_paper.sh

# tables, paired confidence intervals, figures
python scripts/paper_tables.py          # -> results/paper_tables.md
python scripts/paired_ci.py             # -> results/paired_ci.md
python scripts/paper_figures.py         # -> figures/paper/
```

The full run takes many hours on a single consumer GPU/CPU machine (the paper's runs used a
12-thread Ryzen 6600H and an RTX 3050). Pre-computed outputs are in `results/`, so you can
regenerate every table and figure without re-running anything.

`paper_figures.py` takes optional figure numbers (`python scripts/paper_figures.py 3 5`; default all 1 to 7)
and reads only `results/*.json` and `results/paper_tables.md`, so run `PART=T bash scripts/run_paper.sh`
first. `paired_ci.py` takes no arguments.

## Mechanism checks and analysis scripts

| Script | Purpose |
|---|---|
| `scripts/mechanism.py` | Altitude-fix, noise-replay and geometry mechanism tests (`fix`, `noise`, `geom`) |
| `scripts/decision_headroom.py` | First-decision action sweep and hindsight headroom |
| `scripts/fit_decision_policy.py` | Offline conditional-logit chooser |
| `scripts/train_detour.py`, `scripts/eval_detour.py` | Decision-layer PPO training and evaluation |
| `scripts/train_ppo.py`, `scripts/evaluate.py` | Residual PPO training and evaluation |
| `scripts/diagnose_failures.py` | Failure categorisation |
| `scripts/sweep_init.py`, `scripts/sweep_trn_ambiguity.py` | Initial-error and ambiguity sweeps |

## Scope and limits

3-DoF dynamics; terrain is mostly interpolated from sparse LOLA tracks; the baseline is self-built;
the realism-layer values (6 m registration offset, 1 m correlated map error, 40 degree field of
view) are our assumptions, not parameters of a specific instrument; few training seeds. See the
paper for what was measured and what is hypothesis.

## Citation

If you use this code, please cite the archived release (see `CITATION.cff`):

    Chauhan, N. lunar-trnrl: real-terrain lunar landing benchmark with terrain-relative
    navigation. Zenodo, https://doi.org/10.5281/zenodo.23497770

TODO: add the paper's BibTeX once the preprint exists.

## Licence

TODO: see `LICENSE`.
