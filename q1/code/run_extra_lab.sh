#!/usr/bin/env bash
# Extra experiments: flood river (compressed travel times), the upstream
# intervention probe, and the linear information probe.
set -x
cd /home/hussain/river_q1
LOG=results/run.log
python3 scripts/run_experiments.py flood --device cuda --steps 2500 --seeds 2 2>&1 | tee -a $LOG
python3 scripts/upstream_probe.py 2>&1 | tee -a $LOG
python3 scripts/diagnose_spatial.py 2>&1 | tee -a $LOG
python3 scripts/oracle_upstream.py 2>&1 | tee -a $LOG
echo "EXTRA DONE" | tee -a $LOG
