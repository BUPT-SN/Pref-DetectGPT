#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7

DATA_DIR="datasets/Visual"

# 递归遍历 Visual 目录及其子目录中的所有 JSON 数据集
shopt -s globstar nullglob
DATASETS=("$DATA_DIR"/**/*.json)

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "No JSON datasets found under $DATA_DIR"
    exit 1
fi

# TIMESTAMP=$(date +"%Y%m%d_%H%M")
# BASE_RESULTS_DIR="./results_${TIMESTAMP}"

BASE_RESULTS_DIR="./results/Visual"


echo "All results will be saved to: $BASE_RESULTS_DIR"

MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"


echo "Running method: Pref-DetectGPT"
for FILE in "${DATASETS[@]}"; do
    # 1. 获取不带路径的文件名 (例如: multi_domains_arxiv_test.json)
    filename=$(basename "$FILE")
    domain_name=$(basename "$(dirname "$FILE")")
    
    # 2. 去掉 .json 后缀 (例如: multi_domains_arxiv_test)
    clean_name="${filename%.json}"
    
    echo "Processing file: $FILE"

    python Pref/eval_Pref_DetectGPT.py \
        --eval_data_path "$FILE" \
        --eval_data_format ImBD \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/$domain_name/Pref-DetectGPT" \
        --save_file "Pref_${clean_name}.json" \
        --reference_name_or_path "$MODEL_BASE" \
        --policy_name_or_path "$MODEL_INSTRUCT" \
        --sample_name_or_path "$MODEL_BASE"
done
