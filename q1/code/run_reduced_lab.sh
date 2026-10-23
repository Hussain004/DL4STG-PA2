#!/usr/bin/env bash
# Reduced sweep for the remaining ablations (1 seed, fewer steps).
set -x
cd /home/hussain/river_q1
LOG=results/run.log
python3 scripts/run_experiments.py inputs --device cuda --steps 2000 --seeds 1 2>&1 | tee -a $LOG
python3 scripts/run_experiments.py robust --device cuda --steps 2000 --seeds 1 2>&1 | tee -a $LOG
python3 scripts/run_experiments.py loss   --device cuda --steps 2000 --seeds 1 2>&1 | tee -a $LOG
echo "REDUCED DONE" | tee -a $LOG
