#!/usr/bin/env bash
set -euo pipefail

EPOCH=3
BATCH_SIZE=32
LR=5e-6
TRAIN_EMB=true
TRAIN_IMG_PROJ=true
ADD_IMG_TOKEN=true
ADD_SCENE_TOKEN=false
NO_OBJ=false
INPUT_DIM=1024
BIDIRECTION=false
DIFFERENT_LR=false
MAX_OBJ_NUM=100
LORA_R=16
LORA_ALPHA=16
ADD_POS_EMB=false
FEAT_FUSION=false
FUSE_WITH_ID=false
MAX_GRAD_NORM=0.01
SEED=42
USE_LOCATION_TOKEN=false

LLAMA_MODEL_PATH="llm/vicuna-7b-v1.5"
TRAIN_TAG="scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref"
VAL_TAG="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"

EVALUATE=false
DEBUG=false
ENABLE_WANDB=false
GPU_NUM=2
DO_SAVE=true
OTHER_INFO="descrip3d"

PRETRAINED_PATH="" # "./ckpts/ckpt_02.pth"
CONFIG=""
OUTPUT_DIR="outputs/$(date +%Y%m%d_%H%M%S)_lr${LR}_ep${EPOCH}_${TRAIN_TAG}__${VAL_TAG}__${OTHER_INFO}"

mkdir -p "${OUTPUT_DIR}"

export MASTER_ADDR=localhost
export MASTER_PORT=$((54000 + RANDOM % 10000))
export PYTHONPATH="$PYTHONPATH:$(which python):."
echo "PYTHONPATH: $PYTHONPATH"

NNODES=${SLURM_JOB_NUM_NODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
GPUS_PER_NODE=2
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))


echo "Launching training with torchrun..."
torchrun \
  --nproc_per_node=${GPUS_PER_NODE} \
  tasks/train.py \
  "${CONFIG}config.py" \
  output_dir "${OUTPUT_DIR}" \
  scheduler.epochs "${EPOCH}" \
  optimizer.lr "${LR}" \
  model.add_scene_token "${ADD_SCENE_TOKEN}" \
  model.add_img_token "${ADD_IMG_TOKEN}" \
  pretrained_path "${PRETRAINED_PATH}" \
  evaluate "${EVALUATE}" \
  wandb.enable "${ENABLE_WANDB}" \
  gpu_num "${GPU_NUM}" \
  distributed true \
  do_save "${DO_SAVE}" \
  batch_size "${BATCH_SIZE}" \
  model.train_emb "${TRAIN_EMB}" \
  model.train_img_proj "${TRAIN_IMG_PROJ}" \
  train_tag "${TRAIN_TAG}" \
  val_tag "${VAL_TAG}" \
  model.no_obj "${NO_OBJ}" \
  model.input_dim "${INPUT_DIM}" \
  model.bidirection "${BIDIRECTION}" \
  optimizer.different_lr.enable "${DIFFERENT_LR}" \
  model.max_obj_num "${MAX_OBJ_NUM}" \
  lora.lora_r "${LORA_R}" \
  lora.lora_alpha "${LORA_ALPHA}" \
  model.add_pos_emb "${ADD_POS_EMB}" \
  model.feat_fusion "${FEAT_FUSION}" \
  optimizer.max_grad_norm "${MAX_GRAD_NORM}" \
  seed "${SEED}" \
  model.fuse_with_id "${FUSE_WITH_ID}" \
  model.llama_model_path "${LLAMA_MODEL_PATH}" \
  model.use_location_token "${USE_LOCATION_TOKEN}"
