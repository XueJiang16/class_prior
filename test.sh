#!/usr/bin/env bash

METHOD=$1
OUT_DATA=$2
CKPT=$3
OUT_DIR=$4
ID_CLS=$5
SAMPLE_A=$6


python3 -m torch.distributed.launch --nproc_per_node=2 --master_port='29501' test_ood.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir dataset/id_data/imagenet_val \
--out_datadir dataset/ood_data/${OUT_DATA} \
--model_path ${CKPT}  \
--batch 256 \
--logdir ${OUT_DIR} \
--score ${METHOD} \
--id_cls ${ID_CLS} \
--sample_a ${SAMPLE_A} ${@:7}
