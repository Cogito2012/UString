#!/bin/bash
set -x
set -e

# source activate py37
PHASE=$1
EPOCH=$2

LOG_DIR="./logs"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi

LOG="${LOG_DIR}/${PHASE}_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# experiments on DAD dataset
case ${PHASE} in
  train)
    CUDA_VISIBLE_DEVICES="0" python GCRNN_accident.py \
      --dataset dad \
      --feature_name vgg16 \
      --phase train \
      --base_lr 0.001 \
      --batch_size 64 \
      --epoch 200  \
      --test_iter 20 \
      --loss_weight 0.1 \
      --hidden_dim 128 \
      --latent_dim 64 \
      --feature_dim 4096 \
      --gpus "0" \
      --output_dir ./output_dev/vgg16
    ;;
  test)
    CUDA_VISIBLE_DEVICES="0" python GCRNN_accident.py \
      --dataset dad \
      --feature_name vgg16 \
      --phase test \
      --batch_size 16 \
      --hidden_dim 128 \
      --latent_dim 64 \
      --feature_dim 4096 \
      --evaluate_all \
      --visualize \
      --output_dir ./output_dev/vgg16 \
      --model_file ./output_dev/vgg16/dad/snapshot/gcrnn_model_${EPOCH}.pth
    ;;
  *)
    echo "Invalid argument!"
    exit
    ;;
esac
  
echo 'Done!'
