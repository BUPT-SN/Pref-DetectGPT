#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

DATA_DIR="datasets/Attack"

# 数据集列表 (只需要修改这里即可应用到所有方法)
DATASETS=(
    "data_mixing_attacks_test.json"
    "paraphrase_attacks_llm_test.json"
    "perturbation_attacks_llm_test.json"
    "prompt_attacks_llm_test.json"
)

# TIMESTAMP=$(date +"%Y%m%d_%H%M")
# BASE_RESULTS_DIR="./results_${TIMESTAMP}"

BASE_RESULTS_DIR="./results/attack"

echo "All results will be saved to: $BASE_RESULTS_DIR"

MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"

echo "Running method: RoBERTa"
ROBERTA_CONFIGS=(
    "ckpt/roberta_base_ai_detector/last|roberta-base|roberta"
    "ckpt/roberta_large_ai_detector/last|roberta-large|roberta"
)

for CONFIG in "${ROBERTA_CONFIGS[@]}"; do
    # 解析配置字符串
    IFS="|" read -r MODEL_PATH SAVE_DIR SAVE_PREFIX <<< "$CONFIG"

    echo "Running RoBERTa: $SAVE_DIR"

    for FILE in "${DATASETS[@]}"; do
        filename=$(basename "$FILE")
        clean_name="${filename%.json}"
        python other_method/eval_roberta.py \
            --model_name "$MODEL_PATH" \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/$SAVE_DIR" \
            --save_file "${SAVE_PREFIX}_${clean_name}.json" \
            --eval_data_format ImBD
    done
done


export WANDB_MODE=offline
echo "Running method: ImBD"

for FILE in "${DATASETS[@]}"; do

    filename=$(basename "$FILE")
    
    # 2. 去掉 .json 后缀 (例如: multi_domains_arxiv_test)
    clean_name="${filename%.json}"
    python other_method/eval_imbd.py \
        --pretrained_model_name_or_path ckpt/SPO_score_Llama-3.2-1B-Instruct_ref_None_ai_detection_500_polish.raw_data_lr0.0001_bs1_beta0.05_r8_e5 \
        --eval_data_path "$DATA_DIR/$FILE" \
        --wandb False \
        --train_method SPO \
        --eval_batch_size 8 \
        --save_dir $BASE_RESULTS_DIR/ImBD \
        --save_file "ImBD_${clean_name}.json" \
        --eval_data_format ImBD
done


export WANDB_MODE=offline
echo "Running method: DDL"
for FILE in "${DATASETS[@]}"; do
    filename=$(basename "$FILE")
    clean_name="${filename%.json}"
    python other_method/eval_imbd.py \
        --pretrained_model_name_or_path ckpt/DDL_score_Llama-3.2-1B-Instruct_ref_None_ai_detection_500_polish.raw_data_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e5 \
        --eval_data_path "$DATA_DIR/$FILE" \
        --wandb False \
        --train_method DDL \
        --eval_batch_size 8 \
        --save_dir $BASE_RESULTS_DIR/DetectAnyLLM \
        --save_file "DetectAnyLLM_${clean_name}.json" \
        --eval_data_format ImBD
done



echo "Running method: Biscope"
for FILE in "${DATASETS[@]}"; do
    filename=$(basename "$FILE")
    clean_name="${filename%.json}"
    python other_method/biscope.py \
        --eval_data_path "$DATA_DIR/$FILE" \
        --mode eval \
        --model_file ckpt/biscope_classifier.pkl \
        --detect_model "Llama-3.2-1B-Instruct" \
        --data_format ImBD \
        --save_path "$BASE_RESULTS_DIR/BiScope" \
        --save_file "biscope_${clean_name}.json"
done


echo "Running method: Pref-DetectGPT"
for FILE in "${DATASETS[@]}"; do
    # 1. 获取不带路径的文件名 (例如: multi_domains_arxiv_test.json)
    filename=$(basename "$FILE")
    
    # 2. 去掉 .json 后缀 (例如: multi_domains_arxiv_test)
    clean_name="${filename%.json}"
    
    echo "Processing file: $FILE"

    python Pref/eval_Pref_DetectGPT.py \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_data_format ImBD \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Pref-DetectGPT" \
        --save_file "Pref_${clean_name}.json" \
        --reference_name_or_path "$MODEL_BASE" \
        --policy_name_or_path "$MODEL_INSTRUCT" \
        --sample_name_or_path "$MODEL_BASE"
done

# ---------------- 4. Fast-DetectGPT ----------------
echo "Running method: Fast-DetectGPT"

for FILE in "${DATASETS[@]}"; do    
    filename=$(basename "$FILE")
    clean_name="${filename%.json}"
    python other_method/eval_fast_det_gpt.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --discrepancy_analytic \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Fast-DetectGPT" \
        --save_file "FastDetectGPT_${clean_name}.json" \
        --eval_data_format ImBD
done

echo "Running method: IRM"
for FILE in "${DATASETS[@]}"; do
    # 1. 获取不带路径的文件名 (例如: multi_domains_arxiv_test.json)
    filename=$(basename "$FILE")
    
    # 2. 去掉 .json 后缀 (例如: multi_domains_arxiv_test)
    clean_name="${filename%.json}"

    echo "Processing file: $FILE"
    
    python other_method/eval_IRM.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/IRM" \
        --save_file "IRM_${clean_name}.json" \
        --eval_data_format ImBD
done


echo "Running method: dnadetectllm"

for FILE in "${DATASETS[@]}"; do

    filename=$(basename "$FILE")
    
    clean_name="${filename%.json}"

    python other_method/eval_dnadetectllm.py \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_data_format ImBD \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/DNA-DetectLLM" \
        --save_file "dnadetectllm_${clean_name}.json" \
        --observer_name_or_path "$MODEL_BASE" \
        --performer_name_or_path "$MODEL_INSTRUCT"
done

# ---------------- 1. 基础指标 (Likelihood, Logrank, Entropy) ----------------
# 这三个方法使用相同的脚本，只是 --method 参数不同
for METHOD in likelihood logrank entropy; do
    echo "Running method: $METHOD"
    
    for FILE in "${DATASETS[@]}"; do
        filename=$(basename "$FILE")
        clean_name="${filename%.json}"
        python other_method/eval_baselines.py \
            --scoring_model_name "$MODEL_INSTRUCT" \
            --eval_data_path "$DATA_DIR/$FILE" \
            --eval_batch_size 8 \
            --method "$METHOD" \
            --save_path "$BASE_RESULTS_DIR/$METHOD" \
            --save_file "${METHOD}_${clean_name}.json" \
            --eval_data_format ImBD
    done
done

# # # # ---------------- 2. LRR ----------------
echo "Running method: LRR"

for FILE in "${DATASETS[@]}"; do    
    filename=$(basename "$FILE")
    clean_name="${filename%.json}"
    python other_method/eval_lrr.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/LRR" \
        --save_file "LRR_${clean_name}.json" \
        --eval_data_format ImBD
done



# # # # ---------------- 3. Binoculars ----------------
echo "Running method: Binoculars"

for FILE in "${DATASETS[@]}"; do

    # 1. 获取不带路径的文件名 (例如: multi_domains_arxiv_test.json)
    filename=$(basename "$FILE")
    
    # 2. 去掉 .json 后缀 (例如: multi_domains_arxiv_test)
    clean_name="${filename%.json}"

    echo "Processing file: $FILE"
    python other_method/eval_binoculars.py \
        --observer_model_path "$MODEL_BASE" \
        --performer_model_path "$MODEL_INSTRUCT" \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Binoculars" \
        --save_file "Bino_${clean_name}.json" \
        --eval_data_format ImBD
done


# # # ---------------- Lastde++ ----------------
for FILE in "${DATASETS[@]}"; do 
    echo "Running Lastde++ on file: $FILE"
    filename=$(basename "$FILE")
    clean_name="${filename%.json}"

    python other_method/Lastde-Test/lastde_doubleplus.py  \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --eval_data_path "$DATA_DIR/$FILE" \
        --eval_batch_size 1 \
        --save_path "$BASE_RESULTS_DIR/Lastde++" \
        --save_file "Lastde++_${clean_name}.json" \
        --eval_data_format ImBD
done
