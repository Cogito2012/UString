#!/bin/bash
set -x
set -e

source activate py37
PHASE=$1
GPUS=$2
EPOCH=$3
FEATURE=vgg16

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
    CUDA_VISIBLE_DEVICES=$GPUS python GCRNN_accident.py \
      --dataset dad \
      --feature_name $FEATURE \
      --phase train \
      --base_lr 0.0001 \
      --batch_size 10 \
      --epoch $EPOCH \
      --test_iter 64 \
      --hidden_dim 256 \
      --latent_dim 256 \
      --gpus $GPUS \
      --output_dir ./output_dev/gcrnn_auxloss/$FEATURE
    ;;
  test)
    CUDA_VISIBLE_DEVICES=$GPUS python GCRNN_accident.py \
      --dataset dad \
      --feature_name $FEATURE \
      --phase test \
      --batch_size 10 \
      --hidden_dim 256 \
      --latent_dim 256 \
      --gpus $GPUS \
      --visualize \
      --output_dir ./output_dev/gcrnn_auxloss/$FEATURE \
      --model_file ./output_dev/gcrnn_auxloss/$FEATURE/dad/snapshot/gcrnn_model_${EPOCH}.pth
    ;;
  *)
    echo "Invalid argument!"
    exit
    ;;
esac
  
echo 'Done!'
