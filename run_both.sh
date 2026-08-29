#!/bin/bash
set -e
cd /Users/dylangehl/illation-nlp
echo "=== BASELINE ==="
./venv/bin/python train.py --mode baseline --steps 2500 --log_every 50 --eval_every 250 --out runs/baseline
echo "=== ILLATION ==="
./venv/bin/python train.py --mode illation --steps 2500 --log_every 50 --eval_every 250 --out runs/illation
echo "=== DONE ==="
