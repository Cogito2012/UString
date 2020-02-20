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

## experiments on A3D dataset
# python VGRNN_accident.py --dataset a3d --batch_size 16  --epoch 200  --test_iter 40 --output_dir ./output2

# experiments on DAD dataset
case ${PHASE} in
  train)
    CUDA_VISIBLE_DEVICES="2,3" python VGRNN_accident.py \
      --dataset dad \
      --feature_name i3d \
      --phase train \
      --base_lr 0.005 \
      --batch_size 64 \
      --epoch 200  \
      --test_iter 20 \
      --loss_weight 0.1 \
      --hidden_dim 128 \
      --latent_dim 64 \
      --feature_dim 2048 \
      --gpus "2,3" \
      --output_dir ./output_i3d
    ;;
  test)
    CUDA_VISIBLE_DEVICES="0" python VGRNN_accident.py \
      --dataset dad \
      --feature_name i3d \
      --phase test \
      --batch_size 16 \
      --hidden_dim 128 \
      --latent_dim 64 \
      --feature_dim 2048 \
      --evaluate_all \
      --visualize \
      --output_dir ./output_i3d \
      --model_file ./output_i3d/dad/snapshot/vgrnn_model_${EPOCH}.pth
    ;;
  *)
    echo "Invalid argument!"
    exit
    ;;
esac
  
echo 'Done!'
