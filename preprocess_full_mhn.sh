#!/usr/bin/env bash
set -euo pipefail

DATASET="FB15K_DB15K"
DATASET_DIR_NAME="FBDB15K"
DATA_DIR="data/mmkg/${DATASET_DIR_NAME}/norm"
OUTPUT_DIR="data/noisy/${DATASET}_Full_MHN_20"
NOISE_RATIO="0.2"
MODALITIES="img,att"
SEED="2026"
KG_SIDE="both"
AA_SIDE="right"

python src/preprocess_full_mhn.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --noise_ratio "${NOISE_RATIO}" \
  --modalities "${MODALITIES}" \
  --seed "${SEED}" \
  --kg_side "${KG_SIDE}" \
  --aa_side "${AA_SIDE}" \
  --overwrite
