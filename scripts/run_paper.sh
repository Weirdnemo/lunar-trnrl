#!/usr/bin/env bash
# Run every experiment behind the paper tables, in parts, then build the tables.
#   PART=all|A|B|C|D|E|R|L|H|X|S|S2|T  N=200 NB=100 JOBS=2 HAW=/path/Haworth.tif  bash scripts/run_paper.sh
#   QUICK=1 bash scripts/run_paper.sh      # tiny smoke test of every command (a few minutes)
# Parts: A default regime + ceiling   B initial-error sweep   C failure diagnosis   D residual PPO seeds
#        E learned hold/detour seeds   T collect tables (results/paper_tables.md)
# Every job logs to logs_paper/<tag>.log and writes results/<tag>.json (tags start with P_).
set -u
cd "$(dirname "$0")/.."
PART=${PART:-all}; N=${N:-200}; NB=${NB:-100}; NC=${NC:-60}; JOBS=${JOBS:-2}
HAW=${HAW:-$PWD/data/Haworth_final_adj_5mpp_surf.tif}
S04=data/site04.npy; S23=data/site23.npy
STEPS_PPO=${STEPS_PPO:-600000}; STEPS_DET=${STEPS_DET:-6000}
SEEDS=${SEEDS:-"1 2"}                       # seed 0 of the residual PPO runs already exists from E018/E020
if [ "${QUICK:-0}" = 1 ]; then N=3; NB=2; NC=2; STEPS_PPO=2000; STEPS_DET=300; SEEDS="1"; fi
PY=${PY:-python3}
RF=${RF:---realistic}     # realism preset for A/B/C/R: fov 40 deg footprint, 6 m map registration offset (1 sigma/axis), 1 m correlated map error. ASSUMED values, swept in part R
mkdir -p results logs_paper
J=$(mktemp)
job(){ tag=$1; shift; echo "$PY $* > logs_paper/$tag.log 2>&1" >> "$J"; }
go(){ echo "== running $(wc -l < "$J") jobs, $JOBS at a time"; xargs -d '\n' -P "$JOBS" -I{} sh -c '{}' < "$J"; : > "$J"; }
want(){ [ "$PART" = all ] || [ "$PART" = "$1" ]; }

if want A; then
  for t in site04 site23; do
    for pol in baseline hold; do
      job P_A_${t}_${pol}        scripts/eval_detour.py data/$t.npy $pol $N P_A_${t}_${pol} $RF
      job P_A_${t}_${pol}_fixed  scripts/eval_detour.py data/$t.npy $pol $N P_A_${t}_${pol}_fixed $RF --alt_sync
    done
    job P_A_${t}_oracle scripts/eval_detour.py data/$t.npy baseline $N P_A_${t}_oracle $RF --oracle
  done
  for sp in test train; do
    for pol in baseline wait:70; do
      pn=${pol%%:*}
      job P_A_haw_${pn}_${sp}       scripts/eval_windows.py "$HAW" $pol $N P_A_haw_${pn}_${sp} $sp 24 $RF
      job P_A_haw_${pn}_${sp}_fixed scripts/eval_windows.py "$HAW" $pol $N P_A_haw_${pn}_${sp}_fixed $sp 24 $RF --alt_sync
    done
    job P_A_haw_oracle_${sp} scripts/eval_windows.py "$HAW" baseline $N P_A_haw_oracle_${sp} $sp 24 $RF --oracle
  done
  go
fi

if want B; then
  for t in site04 site23; do
    for s in 300 600 1000; do
      job P_B_${t}_${s}_narrow  scripts/eval_detour.py data/$t.npy baseline $NB P_B_${t}_${s}_narrow  $RF --init_pos_h $s --search_cells 20
      job P_B_${t}_${s}_wide    scripts/eval_detour.py data/$t.npy baseline $NB P_B_${t}_${s}_wide    $RF --init_pos_h $s
      job P_B_${t}_${s}_sync    scripts/eval_detour.py data/$t.npy baseline $NB P_B_${t}_${s}_sync    $RF --init_pos_h $s --alt_sync
      job P_B_${t}_${s}_sync_fov60 scripts/eval_detour.py data/$t.npy baseline $NB P_B_${t}_${s}_sync_fov60 $RF --init_pos_h $s --alt_sync --fov 60
      job P_B_${t}_${s}_sync_hold scripts/eval_detour.py data/$t.npy hold $NB P_B_${t}_${s}_sync_hold $RF --init_pos_h $s --alt_sync
    done
  done
  go
fi

if want C; then
  for s in 600 1000; do
    job P_C_diag_${s}_legacy  scripts/diagnose_failures.py $S23 $s $NC P_C_diag_${s}_legacy baseline $RF
    job P_C_diag_${s}_sync    scripts/diagnose_failures.py $S23 $s $NC P_C_diag_${s}_sync baseline $RF --alt_sync
    job P_C_diag_${s}_sync_fov60  scripts/diagnose_failures.py $S23 $s $NC P_C_diag_${s}_sync_fov60 baseline $RF --alt_sync --fov 60
  done
  go
fi

if want D; then
  # residual PPO on the synthetic low-relief benchmark (decoupled filter), extra seeds
  for v in nocov cov; do for sd in $SEEDS; do
    job ppo_${v}_dec_s${sd} scripts/train_ppo.py $v $sd $STEPS_PPO mixed_hard+dec ppo_${v}_dec_s${sd}
  done; done
  # residual PPO on random Haworth windows (default settings), extra seeds
  for sd in $SEEDS; do
    job P_D_haw_nocov_s${sd} scripts/train_ppo_windows.py "$HAW" nocov $sd $STEPS_PPO 48 P_D_haw_nocov_s${sd} --ckpt_every $((STEPS_PPO/20))
  done
  go
  rm -f results/P_D_haw_nocov_s*_step*.zip          # intermediate checkpoints; keep only best/final
  job P_D_dec_baseline scripts/evaluate.py baseline $N P_D_dec_baseline mixed_hard+dec test
  job P_D_dec_hold     scripts/evaluate.py wait:70 $N P_D_dec_hold mixed_hard+dec test
  for v in nocov cov; do for sd in 0 $SEEDS; do
    [ -f results/ppo_${v}_dec_s${sd}.zip ] && job P_D_dec_${v}_s${sd} scripts/evaluate.py ppo:results/ppo_${v}_dec_s${sd}.zip:$v $N P_D_dec_${v}_s${sd} mixed_hard+dec test
  done; done
  for sd in $SEEDS; do for ck in best final; do
    z=results/P_D_haw_nocov_s${sd}_best.zip; [ $ck = final ] && z=results/P_D_haw_nocov_s${sd}.zip
    [ -f "$z" ] && job P_D_haw_nocov_s${sd}_${ck} scripts/eval_windows.py "$HAW" ppo:$z:nocov $N P_D_haw_nocov_s${sd}_${ck} test 24
  done; done
  go
fi

if want E; then
  for sd in 0 1 2; do
    job P_E_train_s${sd} scripts/train_detour.py mixed_hard+dec $STEPS_DET $sd P_E_hold_s${sd} --init_hold
  done
  go
  for pol in baseline hold holdall rule; do
    job P_E_${pol} scripts/eval_detour.py mixed_hard+dec $pol $N P_E_${pol}
  done
  job P_E_oracle scripts/eval_detour.py mixed_hard+dec baseline $N P_E_oracle --oracle
  for sd in 0 1 2; do
    [ -f results/P_E_hold_s${sd}.zip ] && job P_E_learned_s${sd} scripts/eval_detour.py mixed_hard+dec ppo:results/P_E_hold_s${sd}.zip $N P_E_learned_s${sd}
  done
  go
fi

if want R; then
  # (1) realism sensitivity, one knob at a time around the preset (site23, baseline, sync), default error and sigma=600
  for sg in 120 600; do
    for f in 20 30 60; do job P_R_fov${f}_s${sg} scripts/eval_detour.py data/site23.npy baseline $NB P_R_fov${f}_s${sg} --realistic --fov $f --init_pos_h $sg --alt_sync; done
    for o in 0 3 12; do job P_R_off${o}_s${sg} scripts/eval_detour.py data/site23.npy baseline $NB P_R_off${o}_s${sg} --realistic --reg_offset $o --init_pos_h $sg --alt_sync; done
    for m in 0 2 3; do job P_R_noise${m}_s${sg} scripts/eval_detour.py data/site23.npy baseline $NB P_R_noise${m}_s${sg} --realistic --map_noise $m --init_pos_h $sg --alt_sync; done
  done
  go
  # (2) robustness of the nominal-condition RL results to the realism preset (mixed_hard+dec benchmark)
  for pol in baseline hold rule; do job P_R_dec_${pol} scripts/eval_detour.py mixed_hard+dec $pol $N P_R_dec_${pol} --realistic; done
  for sd in 0 1 2; do
    [ -f results/P_E_hold_s${sd}.zip ] && job P_R_dec_learned_s${sd} scripts/eval_detour.py mixed_hard+dec ppo:results/P_E_hold_s${sd}.zip $N P_R_dec_learned_s${sd} --realistic
  done
  go
fi

if want X; then
  # extended Site 23 runs (sigma 600 and 1000, the thin cells in B/R): NX episodes, same seeds 10000+i, so the first 100 equal part B
  NX=${NX:-400}; [ "${QUICK:-0}" = 1 ] && NX=2
  for s in 600 1000; do
    job P_X_site23_${s}_wide      scripts/eval_detour.py data/site23.npy baseline $NX P_X_site23_${s}_wide      $RF --init_pos_h $s
    job P_X_site23_${s}_sync      scripts/eval_detour.py data/site23.npy baseline $NX P_X_site23_${s}_sync      $RF --init_pos_h $s --alt_sync
    job P_X_site23_${s}_sync_fov60 scripts/eval_detour.py data/site23.npy baseline $NX P_X_site23_${s}_sync_fov60 $RF --init_pos_h $s --alt_sync --fov 60
    job P_X_site23_${s}_sync_hold scripts/eval_detour.py data/site23.npy hold $NX P_X_site23_${s}_sync_hold $RF --init_pos_h $s --alt_sync
  done
  for f in 20 30; do job P_X_site23_600_sync_fov${f} scripts/eval_detour.py data/site23.npy baseline $NX P_X_site23_600_sync_fov${f} $RF --init_pos_h 600 --alt_sync --fov $f; done
  go
fi

if want S; then
  # (S) training seeds and ablations answering "too few seeds / throttled residual / warm-start artifact":
  #   residual PPO (synthetic benchmark): seeds 3,4 at k=0.5 for both variants (-> 5 seeds with the earlier 0..2), and 3 seeds at k=1.0 (no cov);
  #   decision layer: 5 independent COLD-START PPO seeds (no warm start), evaluated nominal and under the realism layer,
  #   plus the one-line 'away' rules evaluated on the same episodes.
  STEPS_DETC=${STEPS_DETC:-8000}; [ "${QUICK:-0}" = 1 ] && STEPS_DETC=300
  SUB=${SUB:-all}        # SUB=cold: only the cold-start decision policies and away rules; SUB=resid: only the residual-PPO runs
  if [ "$SUB" != cold ]; then
    for sd in 3 4; do for v in nocov cov; do
      job P_S_${v}_s${sd} scripts/train_ppo.py $v $sd $STEPS_PPO mixed_hard+dec P_S_${v}_s${sd}
    done; done
    for sd in 0 1 2; do
      job P_S_k1_nocov_s${sd} scripts/train_ppo.py nocov $sd $STEPS_PPO mixed_hard+dec P_S_k1_nocov_s${sd} --k 1.0
    done
  fi
  if [ "$SUB" != resid ]; then
    for sd in 0 1 2 3 4; do
      job P_S_cold_s${sd} scripts/train_detour.py mixed_hard+dec $STEPS_DETC $sd P_S_cold_s${sd} --cold --lr 3e-4
    done
  fi
  go
  for sd in 3 4; do for v in nocov cov; do
    [ -f results/P_S_${v}_s${sd}.zip ] && job P_S_dec_${v}_s${sd} scripts/evaluate.py ppo:results/P_S_${v}_s${sd}.zip:$v $N P_S_dec_${v}_s${sd} mixed_hard+dec test
  done; done
  for sd in 0 1 2; do
    [ -f results/P_S_k1_nocov_s${sd}.zip ] && job P_S_dec_k1_nocov_s${sd} scripts/evaluate.py ppo:results/P_S_k1_nocov_s${sd}.zip:nocov@1.0 $N P_S_dec_k1_nocov_s${sd} mixed_hard+dec test
  done
  for sd in 0 1 2 3 4; do
    [ -f results/P_S_cold_s${sd}.zip ] && job P_S_dec_cold_s${sd} scripts/eval_detour.py mixed_hard+dec ppo:results/P_S_cold_s${sd}.zip $N P_S_dec_cold_s${sd}
    [ -f results/P_S_cold_s${sd}.zip ] && job P_S_real_cold_s${sd} scripts/eval_detour.py mixed_hard+dec ppo:results/P_S_cold_s${sd}.zip $N P_S_real_cold_s${sd} --realistic
  done
  for r in away away400; do
    job P_S_dec_${r}  scripts/eval_detour.py mixed_hard+dec $r $N P_S_dec_${r}
    job P_S_real_${r} scripts/eval_detour.py mixed_hard+dec $r $N P_S_real_${r} --realistic
  done
  go
fi

if want S2; then
  # (S2) follow-up to S: (a) the one-line 'away' rules at the FIRST decision only (the policy behind the 93.5% / 84.5% headline;
  # P_S_dec_away* decided at every decision and timed out), (b) the cold-start decision policies SAMPLED from the policy instead of greedy.
  for r in awayfirst away400first; do
    job P_S2_dec_${r}  scripts/eval_detour.py mixed_hard+dec $r $N P_S2_dec_${r}
    job P_S2_real_${r} scripts/eval_detour.py mixed_hard+dec $r $N P_S2_real_${r} --realistic
  done
  for sd in 0 1 2 3 4; do
    [ -f results/P_S_cold_s${sd}.zip ] && job P_S2_dec_sto_s${sd}  scripts/eval_detour.py mixed_hard+dec ppos:results/P_S_cold_s${sd}.zip $N P_S2_dec_sto_s${sd}
    [ -f results/P_S_cold_s${sd}.zip ] && job P_S2_real_sto_s${sd} scripts/eval_detour.py mixed_hard+dec ppos:results/P_S_cold_s${sd}.zip $N P_S2_real_sto_s${sd} --realistic
  done
  go
fi

if want L; then
  # legacy (idealised-map) conditions kept for comparison with E001-E029: fixed 15/31-cell patch, no registration error
  for sg in 120 600; do
    job P_L_s23_${sg}_legacy scripts/eval_detour.py data/site23.npy baseline $NB P_L_s23_${sg}_legacy --init_pos_h $sg
    job P_L_s23_${sg}_sync31 scripts/eval_detour.py data/site23.npy baseline $NB P_L_s23_${sg}_sync31 --init_pos_h $sg --alt_sync --patch_n 31
  done
  go
fi

if want H; then
  # decision-layer headroom + offline-learned first-decision chooser (mixed_hard+dec). ~25-35 min per 200 episodes per core;
  # NH test episodes, NHT train episodes. Realism variants are slower (about 2x).
  NH=${NH:-200}; NHT=${NHT:-400}
  if [ "${QUICK:-0}" = 1 ]; then NH=3; NHT=4; fi
  HNOM=${HNOM:-1}        # HNOM=0: skip the nominal jobs (use results/P_H_*nominal.json copied from elsewhere)
  if [ "$HNOM" = 1 ]; then
    job P_H_test_nominal   scripts/decision_headroom.py mixed_hard+dec $NH P_H_test_nominal
    job P_H_train_nominal  scripts/decision_headroom.py mixed_hard+dec $NHT P_H_train_nominal --train
    job P_H_rule_first_nominal scripts/eval_rule_first.py mixed_hard+dec $NH P_H_rule_first_nominal
  fi
  if [ "${HREAL:-1}" = 1 ]; then
    job P_H_test_realistic  scripts/decision_headroom.py mixed_hard+dec $NH P_H_test_realistic --realistic
    job P_H_train_realistic scripts/decision_headroom.py mixed_hard+dec $NHT P_H_train_realistic --train --realistic
  fi
  go
  [ -f results/P_H_train_nominal.json ] && [ -f results/P_H_test_nominal.json ] && $PY scripts/fit_decision_policy.py results/P_H_train_nominal.json results/P_H_test_nominal.json P_H_fit_nominal > logs_paper/P_H_fit_nominal.log 2>&1
  if [ "${HREAL:-1}" = 1 ]; then
    $PY scripts/fit_decision_policy.py results/P_H_train_realistic.json results/P_H_test_realistic.json P_H_fit_realistic > logs_paper/P_H_fit_realistic.log 2>&1
  fi
fi

if want T || [ "$PART" = all ]; then
  $PY scripts/paired_ci.py > /dev/null && $PY scripts/paper_tables.py
fi
rm -f "$J"
# a QUICK smoke test must not leave P_* files behind (they would mix with the real run's tables)
if [ "${QUICK:-0}" = 1 ] && [ "$PART" != T ]; then rm -rf results/P_*.json results/P_*.zip logs_paper; echo "QUICK run: smoke-test outputs removed"; fi
