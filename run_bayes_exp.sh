#!/bin/bash
set -x
set -e

source activate py37
PHASE=$1
GPUS=$2
EPOCH=$3

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
    CUDA_VISIBLE_DEVICES=$GPUS python BayesGCRNN_accident.py \
      --dataset dad \
      --feature_name vgg16 \
      --phase train \
      --base_lr 0.0001 \
      --batch_size 10 \
      --epoch $EPOCH \
      --test_iter 64 \
      --loss_weight 0.0001 \
      --hidden_dim 256 \
      --latent_dim 256 \
      --feature_dim 4096 \
      --gpus $GPUS \
      --resume \
      --model_file ./output_dev/bayes_gcrnn/vgg16/dad/snapshot/bayesian_gcrnn_model_14.pth \
      --output_dir ./output_dev/bayes_gcrnn/vgg16
    ;;
  test)
    echo "Not Implemented Yet!"
    ;;
  *)
    echo "Invalid argument!"
    exit
    ;;
esac
  
echo 'Done!'
