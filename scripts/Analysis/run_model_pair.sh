#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

DATA_DIR="datasets/Main"

# 数据集列表 (只需要修改这里即可应用到所有方法)
DATASETS=(
    "DetectRL-MutiDomain.json"
    "HART.json"
    "MIRAGE-DIG-2k.json"
    "MIRAGE-SIG-2k.json"
)

# TIMESTAMP=$(date +"%Y%m%d_%H%M")
# BASE_RESULTS_DIR="./results_${TIMESTAMP}"

BASE_RESULTS_DIR="./results/Analysis/Model-Pair"

# instruct|base|pair_name|pref_batch_size
MODEL_PAIRS=(
    "Qwen/Qwen2.5-1.5B-Instruct|Qwen/Qwen2.5-1.5B|Qwen2.5-1.5B|2"
    "google/gemma-2-2b-it|google/gemma-2-2b|Gemma-2-2B|4"
)

echo "All results will be saved to: $BASE_RESULTS_DIR"

for MODEL_PAIR in "${MODEL_PAIRS[@]}"; do
    IFS="|" read -r MODEL_INSTRUCT MODEL_BASE PAIR_NAME PREF_BATCH_SIZE <<< "$MODEL_PAIR"

    echo "=================================================="
    echo "Running model pair: $MODEL_INSTRUCT + $MODEL_BASE"

    echo "Running method: Pref-DetectGPT"
    for FILE in "${DATASETS[@]}"; do
        filename=$(basename "$FILE")
        clean_name="${filename%.json}"

        echo "Processing file: $FILE"

        python Pref/eval_Pref_DetectGPT.py \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_data_format ImBD \
            --eval_batch_size "$PREF_BATCH_SIZE" \
            --save_path "$BASE_RESULTS_DIR/Pref-DetectGPT" \
            --save_file "Pref_${clean_name}_${PAIR_NAME}.json" \
            --reference_name_or_path "$MODEL_BASE" \
            --policy_name_or_path "$MODEL_INSTRUCT" \
            --sample_name_or_path "$MODEL_BASE"
    done

    echo "Running method: IRM"
    for FILE in "${DATASETS[@]}"; do
        filename=$(basename "$FILE")
        clean_name="${filename%.json}"

        echo "Processing file: $FILE"

        python other_method/eval_IRM.py \
            --scoring_model_name "$MODEL_INSTRUCT" \
            --reference_model_name "$MODEL_BASE" \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/IRM" \
            --save_file "IRM_${clean_name}_${PAIR_NAME}.json" \
            --eval_data_format ImBD
    done

    echo "Running method: DNA-DetectLLM"
    for FILE in "${DATASETS[@]}"; do
        filename=$(basename "$FILE")
        clean_name="${filename%.json}"

        echo "Processing file: $FILE"

        python other_method/eval_dnadetectllm.py \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_data_format ImBD \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/DNA-DetectLLM" \
            --save_file "dnadetectllm_${clean_name}_${PAIR_NAME}.json" \
            --observer_name_or_path "$MODEL_BASE" \
            --performer_name_or_path "$MODEL_INSTRUCT"
    done
done
