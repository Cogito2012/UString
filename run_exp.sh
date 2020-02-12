#!/bin/bash
set -x
set -e

# source activate py37
PHASE=$1

LOG_DIR="./logs"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi

LOG="${LOG_DIR}/${PHASE}_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

## experiments on A3D dataset
# python VGRNN_accident.py --dataset a3d --batch_size 16  --epoch 200  --test_iter 40 --output_dir ./output2

# experiments on DAD dataset
case ${PHASE} in
  train)
    python VGRNN_accident.py \
      --dataset dad \
      --phase train \
      --base_lr 0.005 \
      --batch_size 80 \
      --epoch 500  \
      --test_iter 20 \
      --loss_weight 0.1 \
      --hidden_dim 128 \
      --latent_dim 64 \
      --feature_dim 4096 \
      --gpus "0,1,2,3" \
      --output_dir ./output_h
    ;;
  test)
    python VGRNN_accident.py \
      --dataset dad \
      --phase test \
      --batch_size 16 \
      --hidden_dim 128 \
      --latent_dim 64 \
      --feature_dim 4096 \
      --visualize \
      --output_dir ./output_h \
      --model_file ./output_h/dad/snapshot/vgrnn_model_499.pth
    ;;
  *)
    echo "Invalid argument!"
    exit
    ;;
esac
  
echo 'Done!'
