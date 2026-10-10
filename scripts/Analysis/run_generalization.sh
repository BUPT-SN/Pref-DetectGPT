#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

DATA_DIR="datasets/Main"

# 数据集列表
DATASETS=(
    "DetectRL-MutiDomain.json"
    "HART.json"
    "MIRAGE-DIG-2k.json"
    "MIRAGE-SIG-2k.json"
)


BASE_RESULTS_DIR="./results/Analysis/Generalization"

echo "All results will be saved to: $BASE_RESULTS_DIR"

MODEL_INSTRUCT="meta-llama/Llama-3.2-1B-Instruct"
MODEL_BASE="meta-llama/Llama-3.2-1B"


# ==========================================
# 1. Pref-DetectGPT
# ==========================================
echo "Running method: Pref-DetectGPT"
for BASE_FILE in "${DATASETS[@]}"; do
    base_clean_name="${BASE_FILE%.json}"
    
    GEN_DATASETS_PATHS=""
    for GEN_FILE in "${DATASETS[@]}"; do
        if [ "$GEN_FILE" != "$BASE_FILE" ]; then
            GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
        fi
    done

    echo "Processing Base: $BASE_FILE"
    python Analysis/Genelization/Pref/eval_Pref_DetectGPT.py \
        --base_dataset "$DATA_DIR/$BASE_FILE" \
        --generalization_datasets $GEN_DATASETS_PATHS \
        --eval_data_format ImBD \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Pref-DetectGPT" \
        --save_file "Pref_gen_from_${base_clean_name}.json" \
        --reference_name_or_path "$MODEL_BASE" \
        --policy_name_or_path "$MODEL_INSTRUCT" \
        --sample_name_or_path "$MODEL_BASE"
done


# ==========================================
# 2. Fast-DetectGPT
# ==========================================
echo "Running method: Fast-DetectGPT"
for BASE_FILE in "${DATASETS[@]}"; do    
    base_clean_name="${BASE_FILE%.json}"
    
    GEN_DATASETS_PATHS=""
    for GEN_FILE in "${DATASETS[@]}"; do
        if [ "$GEN_FILE" != "$BASE_FILE" ]; then
            GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
        fi
    done

    python Analysis/Genelization/other_method/eval_fast_det_gpt.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --discrepancy_analytic \
        --base_dataset "$DATA_DIR/$BASE_FILE" \
        --generalization_datasets $GEN_DATASETS_PATHS \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Fast-DetectGPT" \
        --save_file "FastDetectGPT_gen_from_${base_clean_name}.json" \
        --eval_data_format ImBD
done


# # ==========================================
# # 3. IRM
# # ==========================================
echo "Running method: IRM"
for BASE_FILE in "${DATASETS[@]}"; do
    base_clean_name="${BASE_FILE%.json}"
    
    GEN_DATASETS_PATHS=""
    for GEN_FILE in "${DATASETS[@]}"; do
        if [ "$GEN_FILE" != "$BASE_FILE" ]; then
            GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
        fi
    done

    python Analysis/Genelization/other_method/eval_IRM.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --discrepancy_analytic \
        --base_dataset "$DATA_DIR/$BASE_FILE" \
        --generalization_datasets $GEN_DATASETS_PATHS \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/IRM" \
        --save_file "IRM_gen_from_${base_clean_name}.json" \
        --eval_data_format ImBD
done



# # ==========================================
# # 4. DNA-DetectLLM
# # ==========================================
echo "Running method: dnadetectllm"
for BASE_FILE in "${DATASETS[@]}"; do
    base_clean_name="${BASE_FILE%.json}"
    
    GEN_DATASETS_PATHS=""
    for GEN_FILE in "${DATASETS[@]}"; do
        if [ "$GEN_FILE" != "$BASE_FILE" ]; then
            GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
        fi
    done

    python Analysis/Genelization/other_method/eval_dnadetectllm.py \
        --base_dataset "$DATA_DIR/$BASE_FILE" \
        --generalization_datasets $GEN_DATASETS_PATHS \
        --eval_data_format ImBD \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/DNA-DetectLLM" \
        --save_file "dnadetectllm_gen_from_${base_clean_name}.json" \
        --observer_name_or_path "$MODEL_BASE" \
        --performer_name_or_path "$MODEL_INSTRUCT"
done


# # ==========================================
# # 5. 基础指标 (Likelihood, Logrank, Entropy)
# # ==========================================
for METHOD in likelihood logrank entropy; do
    echo "Running method: $METHOD"
    
    for BASE_FILE in "${DATASETS[@]}"; do
        base_clean_name="${BASE_FILE%.json}"
        
        GEN_DATASETS_PATHS=""
        for GEN_FILE in "${DATASETS[@]}"; do
            if [ "$GEN_FILE" != "$BASE_FILE" ]; then
                GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
            fi
        done

        python Analysis/Genelization/other_method/eval_baselines.py \
            --scoring_model_name "$MODEL_INSTRUCT" \
            --base_dataset "$DATA_DIR/$BASE_FILE" \
            --generalization_datasets $GEN_DATASETS_PATHS \
            --eval_batch_size 8 \
            --method "$METHOD" \
            --save_path "$BASE_RESULTS_DIR/$METHOD" \
            --save_file "${METHOD}_gen_from_${base_clean_name}.json" \
            --eval_data_format ImBD
    done
done


# # ==========================================
# # 6. LRR
# # ==========================================
echo "Running method: LRR"
for BASE_FILE in "${DATASETS[@]}"; do    
    base_clean_name="${BASE_FILE%.json}"
    
    GEN_DATASETS_PATHS=""
    for GEN_FILE in "${DATASETS[@]}"; do
        if [ "$GEN_FILE" != "$BASE_FILE" ]; then
            GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
        fi
    done

    python Analysis/Genelization/other_method/eval_lrr.py \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --base_dataset "$DATA_DIR/$BASE_FILE" \
        --generalization_datasets $GEN_DATASETS_PATHS \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/LRR" \
        --save_file "LRR_gen_from_${base_clean_name}.json" \
        --eval_data_format ImBD
done



# # ==========================================
# # 7. Binoculars
# # ==========================================
echo "Running method: Binoculars"
for BASE_FILE in "${DATASETS[@]}"; do
    base_clean_name="${BASE_FILE%.json}"
    
    GEN_DATASETS_PATHS=""
    for GEN_FILE in "${DATASETS[@]}"; do
        if [ "$GEN_FILE" != "$BASE_FILE" ]; then
            GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
        fi
    done

    python Analysis/Genelization/other_method/eval_binoculars.py \
        --observer_model_path "$MODEL_BASE" \
        --performer_model_path "$MODEL_INSTRUCT" \
        --base_dataset "$DATA_DIR/$BASE_FILE" \
        --generalization_datasets $GEN_DATASETS_PATHS \
        --eval_batch_size 8 \
        --save_path "$BASE_RESULTS_DIR/Binoculars" \
        --save_file "Bino_gen_from_${base_clean_name}.json" \
        --eval_data_format ImBD
done


# ==========================================
# 8. Lastde++
# ==========================================
for BASE_FILE in "${DATASETS[@]}"; do 
    echo "Running Lastde++ on Base: $BASE_FILE"
    base_clean_name="${BASE_FILE%.json}"
    
    GEN_DATASETS_PATHS=""
    for GEN_FILE in "${DATASETS[@]}"; do
        if [ "$GEN_FILE" != "$BASE_FILE" ]; then
            GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
        fi
    done

    # 注意：确保 lastde_doubleplus.py 内部也像前面那些脚本一样修改了传参和逻辑
    python Analysis/Genelization/other_method/Lastde-Test/lastde_doubleplus.py  \
        --scoring_model_name "$MODEL_INSTRUCT" \
        --reference_model_name "$MODEL_BASE" \
        --base_dataset "$DATA_DIR/$BASE_FILE" \
        --generalization_datasets $GEN_DATASETS_PATHS \
        --eval_batch_size 1 \
        --save_path "$BASE_RESULTS_DIR/Lastde++" \
        --save_file "Lastde++_gen_from_${base_clean_name}.json" \
        --eval_data_format ImBD
done



# ==========================================
# 1. RoBERTa
# ==========================================
echo "Running method: RoBERTa for Threshold Generalization"
ROBERTA_CONFIGS=(
    "ckpt/roberta_base_ai_detector/last|roberta-base|roberta"
    "ckpt/roberta_large_ai_detector/last|roberta-large|roberta"
)

for CONFIG in "${ROBERTA_CONFIGS[@]}"; do
    IFS="|" read -r MODEL_PATH SAVE_DIR SAVE_PREFIX <<< "$CONFIG"
    echo "-----------------------------------------------------"
    echo "Running RoBERTa: $SAVE_DIR"

    for BASE_FILE in "${DATASETS[@]}"; do
        base_clean_name="${BASE_FILE%.json}"
        
        # 构造泛化数据集列表 (排除当前的 Base Dataset)
        GEN_DATASETS_PATHS=""
        for GEN_FILE in "${DATASETS[@]}"; do
            if [ "$GEN_FILE" != "$BASE_FILE" ]; then
                GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
            fi
        done

        python Analysis/Genelization/other_method/eval_roberta.py \
            --model_name "$MODEL_PATH" \
            --base_dataset "$DATA_DIR/$BASE_FILE" \
            --generalization_datasets $GEN_DATASETS_PATHS \
            --eval_batch_size 8 \
            --save_path "$BASE_RESULTS_DIR/$SAVE_DIR" \
            --save_file "${SAVE_PREFIX}_gen_from_${base_clean_name}.json" \
            --eval_data_format ImBD
    done
done


# # ==========================================
# # 2. ImBD (SPO)
# # ==========================================
export WANDB_MODE=offline
echo "====================================================="
echo "Running method: ImBD"

for BASE_FILE in "${DATASETS[@]}"; do
    base_clean_name="${BASE_FILE%.json}"
    
    GEN_DATASETS_PATHS=""
    for GEN_FILE in "${DATASETS[@]}"; do
        if [ "$GEN_FILE" != "$BASE_FILE" ]; then
            GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
        fi
    done

    python Analysis/Genelization/other_method/eval_imbd.py \
        --pretrained_model_name_or_path ckpt/SPO_score_Llama-3.2-1B-Instruct_ref_None_ai_detection_500_polish.raw_data_lr0.0001_bs1_beta0.05_r8_e5 \
        --base_dataset "$DATA_DIR/$BASE_FILE" \
        --generalization_datasets $GEN_DATASETS_PATHS \
        --wandb False \
        --train_method SPO \
        --eval_batch_size 8 \
        --save_dir "$BASE_RESULTS_DIR/ImBD" \
        --save_file "ImBD_gen_from_${base_clean_name}.json" \
        --eval_data_format ImBD
done


# # ==========================================
# # 3. DetectAnyLLM (DDL)
# # ==========================================
export WANDB_MODE=offline
echo "====================================================="
echo "Running method: DDL (DetectAnyLLM)"

for BASE_FILE in "${DATASETS[@]}"; do
    base_clean_name="${BASE_FILE%.json}"
    
    GEN_DATASETS_PATHS=""
    for GEN_FILE in "${DATASETS[@]}"; do
        if [ "$GEN_FILE" != "$BASE_FILE" ]; then
            GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
        fi
    done

    python Analysis/Genelization/other_method/eval_imbd.py \
        --pretrained_model_name_or_path ckpt/DDL_score_Llama-3.2-1B-Instruct_ref_None_ai_detection_500_polish.raw_data_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e5 \
        --base_dataset "$DATA_DIR/$BASE_FILE" \
        --generalization_datasets $GEN_DATASETS_PATHS \
        --wandb False \
        --train_method DDL \
        --eval_batch_size 8 \
        --save_dir "$BASE_RESULTS_DIR/DetectAnyLLM" \
        --save_file "DetectAnyLLM_gen_from_${base_clean_name}.json" \
        --eval_data_format ImBD
done


# ==========================================
# 4. BiScope
# ==========================================
echo "====================================================="
echo "Running method: Biscope"

for BASE_FILE in "${DATASETS[@]}"; do
    base_clean_name="${BASE_FILE%.json}"
    
    GEN_DATASETS_PATHS=""
    for GEN_FILE in "${DATASETS[@]}"; do
        if [ "$GEN_FILE" != "$BASE_FILE" ]; then
            GEN_DATASETS_PATHS="$GEN_DATASETS_PATHS $DATA_DIR/$GEN_FILE"
        fi
    done

    python Analysis/Genelization/other_method/biscope.py \
        --base_dataset "$DATA_DIR/$BASE_FILE" \
        --generalization_datasets $GEN_DATASETS_PATHS \
        --mode eval \
        --model_file ckpt/biscope_classifier.pkl \
        --detect_model "Llama-3.2-1B-Instruct" \
        --data_format ImBD \
        --save_path "$BASE_RESULTS_DIR/BiScope" \
        --save_file "biscope_gen_from_${base_clean_name}.json"
done

