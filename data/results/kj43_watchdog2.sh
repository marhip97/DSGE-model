#!/bin/bash
LOG=/home/user/DSGE-model/data/results/kj43_nohup.log
WLOG=/home/user/DSGE-model/data/results/kj43_watchdog.log
DONE=/home/user/DSGE-model/data/results/chain_kj43_prod_posterior.json

PY_PID=4623

while true; do
  TS=$(date '+%H:%M:%S')

  # Ferdig?
  if [ -f "$DONE" ]; then
    echo "[$TS] FERDIG — posterior funnet, avslutter watchdog" >> $WLOG
    exit 0
  fi

  # Sjekk om prosessen lever
  if ! ps -p $PY_PID > /dev/null 2>&1; then
    echo "[$TS] DEAD pid=$PY_PID — starter på nytt" >> $WLOG
    cd /home/user/DSGE-model
    python scripts/kj43_prod.py >> $LOG 2>&1 &
    PY_PID=$!
    disown $PY_PID
    echo "[$TS] Ny PID=$PY_PID" >> $WLOG
  fi

  sleep 30
done
