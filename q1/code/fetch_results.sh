#!/usr/bin/env bash
# Copy the experiment results back from the Lab PC.
set -e
LAB=hussain@100.100.67.1
SRC=/home/hussain/DL4TSG/PA2/Question1/investigation
rsync -az --exclude 'cache' "$LAB:/home/hussain/river_q1/results/" "$SRC/results/"
cp -f "$LAB:/home/hussain/river_q1/results/"*.json "$SRC/results/" 2>/dev/null || true
ls -la "$SRC/results/" | head -30
