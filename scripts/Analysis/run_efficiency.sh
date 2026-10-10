#!/bin/bash

# 效率分析：统计各方法的推理速度（每样本平均耗时 ms/sample，换算成 s/1k tokens）
# 1000 个样本，每条精确截断到 100 token，逐样本推理（batch=1），取平均
#
# 和 scripts/Main 保持同样的双卡配置：多模型的方法（Pref / Binoculars / Fast-DetectGPT /
# IRM / DNA-DetectLLM）会把不同模型放到两张卡上，前向可以跨卡重叠。主实验就是这么跑的，
# 测出来的才是实际部署成本；论文表注里写明是双卡即可。
export CUDA_VISIBLE_DEVICES=0,1

DATA_DIR="datasets/Main"
DATASET="HART.json"
BASE_RESULTS_DIR="./results/Analysis/Efficiency"

MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"

METHODS=(
    "RoBERTa-Base"
    "RoBERTa-Large"
    "BiScope"
    "ImBD"
    "DetectAnyLLM"
    "Likelihood"
    "LogRank"
    "Entropy"
    "LRR"
    "Lastde++"
    "Fast-DetectGPT"
    "Binoculars"
    "DNA-DetectLLM"
    "IRM"
    "Pref-DetectGPT"
)

echo "All results will be saved to: $BASE_RESULTS_DIR"

for METHOD in "${METHODS[@]}"; do
    echo "=================================================="
    echo "Running method: $METHOD"

    python scripts/Analysis/eval_efficiency.py \
        --method "$METHOD" \
        --eval_data_path "$DATA_DIR/$DATASET" \
        --eval_data_format ImBD \
        --eval_batch_size 1 \
        --num_docs 1000 \
        --tokens_per_doc 100 \
        --warmup 20 \
        --policy_name_or_path "$MODEL_INSTRUCT" \
        --reference_name_or_path "$MODEL_BASE" \
        --save_path "$BASE_RESULTS_DIR" \
        --save_file "efficiency.csv"
done
