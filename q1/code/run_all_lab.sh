#!/usr/bin/env bash
# Run every Q1 design experiment on the Lab PC (GPU) and keep a log.
set -x
cd /home/hussain/river_q1
LOG=results/run.log
mkdir -p results
run () {  # $1 = experiment, $2 = steps, $3 = seeds
  echo "===================== $1 (steps=$2 seeds=$3) =====================" >> $LOG
  python3 scripts/run_experiments.py "$1" --device cuda --steps "$2" --seeds "$3" 2>&1 | tee -a $LOG
}
run arch      3000 3
run lookback  2500 2
run rounds    2500 2
run inputs    2500 2
run loss      2500 2
run robust    2500 2
run final     3000 3
echo "ALL DONE" | tee -a $LOG
