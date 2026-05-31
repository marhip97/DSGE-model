#!/bin/bash
# Watchdog for kj40: restart kj40_prod.py hvis den dør, stopp når fullført.
cd /home/user/DSGE-model

SCRIPT="scripts/kj40_prod.py"
LOG="data/results/kj40_nohup.log"
DONE_FILE="data/results/chain_kj40_prod_posterior.json"
WATCHDOG_LOG="data/results/kj40_watchdog.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"; }

log "Watchdog startet."

while true; do
    # Ferdig?
    if [ -f "$DONE_FILE" ]; then
        log "kj40 fullført — posterior funnet. Watchdog avslutter."
        exit 0
    fi

    # Kjører?
    if pgrep -f "kj40_prod.py" > /dev/null 2>&1; then
        log "kj40 kjører (PID=$(pgrep -f kj40_prod.py)). OK."
    else
        log "kj40 er nede — restarter..."
        python "$SCRIPT" >> "$LOG" 2>&1 &
        disown $!
        log "kj40 restartet med PID=$!"
    fi

    sleep 60
done
