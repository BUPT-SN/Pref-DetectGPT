#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5

DATA_DIR="datasets/Main"

# 数据集列表 (只需要修改这里即可应用到所有方法)
DATASETS=(
    "HART.json"
    "DetectRL-MutiDomain.json"
    "MIRAGE-DIG-2k.json"
    "MIRAGE-SIG-2k.json"
)

# ====== 新增：Sample 模型列表 ======
SAMPLE_MODELS=(
    "meta-llama/Llama-3-8B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
)

# TIMESTAMP=$(date +"%Y%m%d_%H%M")
# BASE_RESULTS_DIR="./results_${TIMESTAMP}"

BASE_RESULTS_DIR="./results/sample_model"

echo "All results will be saved to: $BASE_RESULTS_DIR"

MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"


echo "Running method: Pref-DetectGPT"
for FILE in "${DATASETS[@]}"; do
    # 1. 获取不带路径的文件名 (例如: multi_domains_arxiv_test.json)
    filename=$(basename "$FILE")
    
    # 2. 去掉 .json 后缀 (例如: multi_domains_arxiv_test)
    clean_name="${filename%.json}"
    
    echo "Processing dataset: $FILE"

    # ====== 新增：遍历 sample 模型 ======
    for SAMPLE_MODEL in "${SAMPLE_MODELS[@]}"; do
        
        # 提取模型名称的最后一部分用于文件命名 (例如提取出 "Llama-3-8B")
        sample_short_name=$(basename "$SAMPLE_MODEL")
        
        echo "  --> Using sample model: $SAMPLE_MODEL"

        python Pref/eval_Pref_DetectGPT.py \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_data_format ImBD \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/Pref-DetectGPT" \
            --save_file "Pref_${clean_name}_${sample_short_name}.json" \
            --reference_name_or_path "$MODEL_BASE" \
            --policy_name_or_path "$MODEL_INSTRUCT" \
            --sample_name_or_path "$SAMPLE_MODEL" # ====== 新增传参 ======
            
    done
done


echo "Running method: Fast-DetectGPT"
for FILE in "${DATASETS[@]}"; do
    # 1. 获取不带路径的文件名 (例如: multi_domains_arxiv_test.json)
    filename=$(basename "$FILE")
    
    # 2. 去掉 .json 后缀 (例如: multi_domains_arxiv_test)
    clean_name="${filename%.json}"
    
    echo "Processing dataset: $FILE"

    # ====== 新增：遍历 sample 模型 ======
    for SAMPLE_MODEL in "${SAMPLE_MODELS[@]}"; do
        
        # 提取模型名称的最后一部分用于文件命名 (例如提取出 "Llama-3-8B")
        sample_short_name=$(basename "$SAMPLE_MODEL")
        
        echo "  --> Using sample model: $SAMPLE_MODEL"

        python other_method/eval_fast_det_gpt.py \
            --scoring_model_name "$MODEL_INSTRUCT" \
            --reference_model_name "$SAMPLE_MODEL" \
            --discrepancy_analytic \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/Fast-DetectGPT" \
            --save_file "FastDetectGPT_${clean_name}_${sample_short_name}.json" \
            --eval_data_format ImBD
            
    done
done