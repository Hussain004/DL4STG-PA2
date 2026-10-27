#!/bin/bash
# Commands used for models B, C, D and D+B (run on the GPU machine from the folder holding the q2/code scripts).
# Stage 1: candidate designs for model D (name patch hidden kernels dilations), trained at speeds 1-3 and
# compared on the validation split.  Da-Dc were the first round (dilated layer 2, dilation 3, 8x8 patches),
# E1-E4 the second (larger kernels, a third layer).  E2 (5x5 kernels, 40 hidden channels) became model D.
mkdir -p runs/logs runs/eval runs/models
for spec in "Da 4 64 3,3 1,2" "Db 4 64 3,3 1,3" "Dc 8 64 3,3 1,1" \
            "E1 4 56 5,3 1,1" "E2 4 40 5,5 1,1" "E3 4 56 3,3,3 1,1,1" "E4 4 48 3,5 1,1"; do
  set -- $spec
  A="--arch reach --patch $2 --hidden $3 --kernels $4 --dilations $5"
  python3 train_models.py --name $1 --seed 0 $A --speed-aug
  python3 evaluate_models.py --ckpt runs/models/$1_seed0.pt $A --split val --out runs/eval/val_$1_seed0.json
done
# Stage 2: model D is candidate "E2".  Models B, C, D, D+B, and A re-trained with the new window sampler
# (Afix), both seeds; then the test-split evaluations (plain and symmetrised).
D_ARGS="--arch reach --patch 4 --hidden 40 --kernels 5,5 --dilations 1,1"
tr() { name=$1; seed=$2; shift 2; python3 train_models.py --name $name --seed $seed "$@"; }
ev() { name=$1; seed=$2; tag=$3; shift 3
       python3 evaluate_models.py --ckpt runs/models/${name}_seed${seed}.pt --out runs/eval/eval_${name}${tag}_seed${seed}.json "$@"; }
for s in 0 1; do
  tr B $s --sym-aug
  tr C $s --speed-aug
  tr Afix $s
  tr D $s $D_ARGS --speed-aug
  tr DB $s $D_ARGS --speed-aug --sym-aug
done
for s in 0 1; do
  python3 evaluate_models.py --ckpt runs/model_a/model_a_seed$s.pt --out runs/eval/eval_A_seed$s.json
  python3 evaluate_models.py --ckpt runs/model_a/model_a_seed$s.pt --wrap --out runs/eval/eval_A_sym_seed$s.json
  ev Afix $s ""
  for m in B C; do ev $m $s ""; ev $m $s _sym --wrap; done
  for m in D DB; do ev $m $s "" $D_ARGS; ev $m $s _sym $D_ARGS --wrap; done
done
# Validation-split evaluations used to choose between D and D+B for the final model.
for s in 0 1; do
  ev C $s _val --split val
  for m in D DB; do ev $m $s _val $D_ARGS --split val; ev $m $s _symval $D_ARGS --wrap --split val; done
done
