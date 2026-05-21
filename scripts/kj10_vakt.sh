#!/bin/bash
# Vakt-skript: starter kjøring 10 på nytt hvis prosessen dør
LOG="data/results/kjoring10_log.txt"
PARTIAL="data/results/chain_kj10_prod_partial.npy"
cd /home/user/DSGE-model

while true; do
    if ! pgrep -f "fase2_kj10_akkumuler" > /dev/null; then
        echo "[$(date)] Starter kjøring 10..." >> "$LOG"
        python3 scripts/fase2_kj10_akkumuler.py >> "$LOG" 2>&1 &
    fi
    sleep 60
done
