#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

DATA_DIR="datasets/Visual/HART_Domain"

# 数据集列表
DATASETS=(
    "arxiv.test.level1.extracted.json"
    "essay.test.level1.extracted.json"
    "news.test.level1.extracted.json"
    "writing.test.level1.extracted.json"
)

PROMPTS=(
    ""
    "You are an academic researcher. Write a concise scientific abstract including background, methodology, results, and conclusions:"
    "You are a student. Write a college application essay with a compelling personal narrative, specific examples, and a clear reflection on your growth and values:"
    "You are a journalist. Write a news article following the inverted pyramid structure with a strong lede, supporting details, and relevant context:"
    "You are a creative writer. Write a short story with a well-developed character, vivid descriptive details, and a satisfying narrative arc:"
)
# 定义 Prompt 的短标识符（用于生成区分度高的保存文件名）
PROMPT_IDS=(
    "p5_empty"
    "p1_academic"
    "p2_student"
    "p3_journalist"
    "p4_creative"
)

BASE_RESULTS_DIR="./results/Analysis"

echo "All results will be saved to: $BASE_RESULTS_DIR"

MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"

echo "Running method: Pref-Prompt"

# 第一层循环：遍历 prompts
for i in "${!PROMPTS[@]}"; do
    PROMPT_TEXT="${PROMPTS[$i]}"
    PROMPT_ID="${PROMPT_IDS[$i]}"
    
    echo "=================================================="
    echo "Running with Prompt [${PROMPT_ID}]: '${PROMPT_TEXT}'"
    
    # 第二层循环：遍历数据集
    for FILE in "${DATASETS[@]}"; do
        filename=$(basename "$FILE")
        clean_name="${filename%.json}"
        
        echo "  -> Processing file: $FILE"

        python Analysis/Pref_prompt/eval_Pref_DetectGPT.py \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_data_format ImBD \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/Pref-Prompt" \
            --save_file "Pref_${clean_name}_${PROMPT_ID}.json" \
            --reference_name_or_path "$MODEL_BASE" \
            --policy_name_or_path "$MODEL_INSTRUCT" \
            --sample_name_or_path "$MODEL_BASE" \
            --prompt "$PROMPT_TEXT"
            
    done
done

echo "All evaluations completed!"