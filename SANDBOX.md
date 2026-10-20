# Landing sandbox and random-window RL

Everything runs on your computer. The big GeoTIFF never has to be uploaded.

## 1. Install (once, in your `trnrl` environment)
    pip install rasterio numpy scipy pillow stable-baselines3 gymnasium

## 2. Try the sandbox with no RL model
    python sandbox/server.py --dem Site04_final_adj_5mpp_surf.tif
Open http://localhost:8000. Click the map to choose an 8 x 8 km window, press Fly.
Standard guidance and the hover-and-wait rule fly the same window, same start, same random events.
Colour the ground by height, safe landing ground (slope under 10 degrees) or where terrain matching works.
"Another start" draws a new random start and target. The right 30% of the map is shaded: RL never trains there.

## 3. Train RL on random windows (left 70% of the map only)
    python scripts/train_ppo_windows.py Site04_final_adj_5mpp_surf.tif cov 0 600000 48 ppo_win_cov_s0
About 10 to 60 minutes on a laptop CPU. It saves a checkpoint every 30k steps (`<tag>_step<N>.zip`) and the best-by-training-score one (`<tag>_best.zip`).
Evaluate several of them (final, best, an early one such as step60000) with eval_windows.py; the final model can be the worst one. Use `nocov` for the no-covariance variant and other seeds (1, 2) for repeats.
The default split needs a map at least 8 km tall and about 27 km wide, so the 30% held-out strip is wider than 8 km.
PGDA site files are usually smaller. Then train on one site and test on a different one (a stronger test anyway):
add `--frac 1.0` to every command, train with Site04's TIFF, and evaluate with Site23's TIFF.

## 4. Test on held-out terrain (right 30%)
    python scripts/eval_windows.py Site04_final_adj_5mpp_surf.tif baseline 200 win_base
    python scripts/eval_windows.py Site04_final_adj_5mpp_surf.tif wait:70 200 win_wait70
    python scripts/eval_windows.py Site04_final_adj_5mpp_surf.tif ppo:results/ppo_win_cov_s0.zip:cov 200 win_ppo_cov_s0
All three use the same windows and the same episode seeds, so the comparison is paired.

## 5. Watch the trained policy
    python sandbox/server.py --dem Site04_final_adj_5mpp_surf.tif --model results/ppo_win_cov_s0.zip --variant cov
