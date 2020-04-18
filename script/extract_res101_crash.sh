#!/bin/bash

source activate py37

GPU_ID=$1
CRASH_DIR=$2

CUDA_VISIBLE_DEVICES=0 time python script/extract_res101_crash.py \
    --crash_dir $CRASH_DIR \
    --imdb voc_2007_train \
    --model lib/frcnn/output/res101/voc_2007_train/default/res101_faster_rcnn_iter_200000.ckpt \
    --cfg lib/frcnn/experiments/cfgs/res101.yml \
    --net res101 \
    --set ANCHOR_SCALES [2,4,8,16,32] ANCHOR_RATIOS [0.33,0.5,1,2,3]