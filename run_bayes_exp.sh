#!/bin/bash
set -x
set -e

source activate py37
PHASE=$1
GPUS=$2
FEATURE=$3
OUT_DIR=$4

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
      --feature_name $FEATURE \
      --phase train \
      --base_lr 0.001 \
      --gpus $GPUS \
      --output_dir ./output_dev/$OUT_DIR/$FEATURE
    ;;
  test)
    CUDA_VISIBLE_DEVICES=$GPUS python BayesGCRNN_accident.py \
      --dataset dad \
      --feature_name $FEATURE \
      --phase test \
      --gpus $GPUS \
      --visualize \
      --output_dir ./output_dev/$OUT_DIR/$FEATURE \
      --model_file ./output_dev/$OUT_DIR/$FEATURE/dad/snapshot/bayesian_gcrnn_model_final.pth
    ;;
  *)
    echo "Invalid argument!"
    exit
    ;;
esac
  
echo 'Done!'
