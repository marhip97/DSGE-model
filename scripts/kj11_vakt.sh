#!/bin/bash
# Vakt-skript: starter kjøring 11 på nytt hvis prosessen dør
LOG="data/results/kjoring11_log.txt"
cd /home/user/DSGE-model

while true; do
    if ! pgrep -f "fase2_kj11_akkumuler" > /dev/null; then
        # Merger temp inn i prod hvis begge finnes
        python3 -c "
import numpy as np
from pathlib import Path
p_path = Path('data/results/chain_kj11_prod_partial.npy')
t_path = Path('data/results/chain_kj11_temp_partial.npy')
if p_path.exists() and t_path.exists():
    p = np.load(str(p_path))
    t = np.load(str(t_path))
    if len(t) > 0:
        c = np.concatenate([p, t])
        np.save(str(p_path), c)
        print(f'MERGED: {len(p)}+{len(t)}={len(c)} trekk')
" >> "$LOG" 2>&1
        echo "[$(date)] Starter kjøring 11..." >> "$LOG"
        nohup python3 scripts/fase2_kj11_akkumuler.py >> "$LOG" 2>&1 &
    fi
    sleep 60
done
