#!/bin/bash
# Kjøres etter fase2v2_production.py er ferdig
set -e
echo "=== Post-prosessering Fase 2v2 ==="
python scripts/d_model_fit_fase2v2.py
echo "=== Ferdig ==="
