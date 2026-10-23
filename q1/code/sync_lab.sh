#!/usr/bin/env bash
# Sync the investigation scripts to the Lab PC and prepare a remote run.
set -euo pipefail
LAB=hussain@100.100.67.1
SRC=/home/hussain/DL4TSG/PA2/Question1/investigation
DST=/home/hussain/river_q1

ssh -o StrictHostKeyChecking=no "$LAB" "mkdir -p $DST/scripts $DST/results $DST/figures"
# never overwrite a script that is currently executing on the remote
rsync -az --exclude '__pycache__' --exclude 'results/cache' \
      --exclude 'run_all_lab.sh' --exclude 'run_extra_lab.sh' \
      "$SRC/scripts/" "$LAB:$DST/scripts/"
rsync -az --exclude 'results/cache' \
      "$SRC/results/" "$LAB:$DST/results/" 2>/dev/null || true
ssh -o StrictHostKeyChecking=no "$LAB" "cd $DST && python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'"
echo "synced to $DST"
