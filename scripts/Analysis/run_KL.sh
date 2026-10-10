#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

DATA_DIR="datasets/Main"

# 数据集列表 (只需要修改这里即可应用到所有方法)
DATASETS=(
    "HART.json"
    "MIRAGE-DIG-2k.json"
    "MIRAGE-SIG-2k.json"
    "DetectRL-MutiDomain.json"
)

# TIMESTAMP=$(date +"%Y%m%d_%H%M")
# BASE_RESULTS_DIR="./results_${TIMESTAMP}"

BASE_RESULTS_DIR="./results/Analysis"


echo "All results will be saved to: $BASE_RESULTS_DIR"

MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"


echo "Running method: Pref-KL"
for FILE in "${DATASETS[@]}"; do
    # 1. 获取不带路径的文件名 (例如: multi_domains_arxiv_test.json)
    filename=$(basename "$FILE")
    
    # 2. 去掉 .json 后缀 (例如: multi_domains_arxiv_test)
    clean_name="${filename%.json}"
    
    echo "Processing file: $FILE"

    python Analysis/Pref_KL/eval_Pref_DetectGPT.py \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_data_format ImBD \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Pref-KL" \
        --save_file "Pref_${clean_name}.json" \
        --reference_name_or_path "$MODEL_BASE" \
        --policy_name_or_path "$MODEL_INSTRUCT" \
        --sample_name_or_path "$MODEL_BASE"
done
