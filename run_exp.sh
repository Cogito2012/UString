#!/bin/bash
set -x
set -e

# source activate py37

LOG_DIR="./logs"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi

LOG="${LOG_DIR}/train_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

## experiments on A3D dataset
# python VGRNN_accident.py --dataset a3d --batch_size 16  --epoch 200  --test_iter 40 --output_dir ./output2

# experiments on DAD dataset
python VGRNN_accident.py \
    --dataset dad \
    --phase train \
    --base_lr 0.0001 \
    --batch_size 80 \
    --epoch 100  \
    --test_iter 20 \
    --loss_weight 0.1 \
    --hidden_dim 128 \
    --latent_dim 64 \
    --feature_dim 4096 \
    --gpus "0,1,2,3" \
    --output_dir ./output_multigpu

## demo
#python VGRNN_accident.py \
#    --dataset dad \
#    --base_lr 0.001 \
#    --batch_size 16 \
#    --epoch 100 \
#    --test_iter 20 \
#    --loss_weight 0.1 \
#    --hidden_dim 128 \
#    --latent_dim 64 \
#    --feature_dim 4096 \
#    --output_dir ./output \
#    --phase test \
#    --model_file ./output/dad/snapshot/vgrnn_model_80.pth


echo 'Done!'
