#!/usr/bin/env bash
set -euo pipefail

: "${OPENVLA_CHECKPOINT:?set OPENVLA_CHECKPOINT}"
: "${DRAFTER_TRAIN_DATA:?set DRAFTER_TRAIN_DATA}"
: "${DRAFTER_OUTPUT_DIR:?set DRAFTER_OUTPUT_DIR}"

GPU_IDS=${GPU_IDS:-0}
MASTER_PORT=${MASTER_PORT:-23333}

WANDB_MODE=${WANDB_MODE:-offline} deepspeed \
  --master_port "${MASTER_PORT}" \
  --include "localhost:${GPU_IDS}" \
  train_deepspeed_libero_goal.py \
  --base_model_path "${OPENVLA_CHECKPOINT}" \
  --data_path "${DRAFTER_TRAIN_DATA}" \
  --output_dir "${DRAFTER_OUTPUT_DIR}" \
  --model_config_path "${DRAFTER_CONFIG_PATH:-llama_2_chat_7B_config.json}" \
  --deepspeed_config "${DEEPSPEED_CONFIG:-ds_config.json}"
