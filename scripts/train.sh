#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

############################################### DetectAnyLLM ##########################################################

TRAIN_METHOD="DDL"
SCORING_MODEL="meta-llama/Llama-3.2-1B-Instruct"
REFERENCE_MODEL="None"

python other_method/train_imbd.py \
    --scoring_model_name ${SCORING_MODEL} \
    --reference_model_name ${REFERENCE_MODEL} \
    --train_data_path datasets/ai_detection_500_polish.raw_data.json \
    --train_data_format ImBD \
    --eval_data_path datasets/ai_detection_500_polish.raw_data.json \
    --save_freq 5 \
    --train_method ${TRAIN_METHOD} \
    --eval_batch_size 4


############################################### ImBD ##########################################################

TRAIN_METHOD="SPO"
SCORING_MODEL="meta-llama/Llama-3.2-1B-Instruct"
REFERENCE_MODEL="None"

python other_method/train_imbd.py \
    --scoring_model_name ${SCORING_MODEL} \
    --reference_model_name ${REFERENCE_MODEL} \
    --train_data_path datasets/ai_detection_500_polish.raw_data.json \
    --train_data_format ImBD \
    --eval_data_path datasets/ai_detection_500_polish.raw_data.json \
    --save_freq 5 \
    --train_method ${TRAIN_METHOD} \
    --eval_batch_size 4


############################################### BiScope ##########################################################

python other_method/biscope.py \
    --mode train \
    --train_data_path datasets/ai_detection_500_polish.raw_data.json \
    --model_file biscope_classifier.pkl \
    --detect_model "Llama-3.2-1B-Instruct" \
    --data_format ImBD \
    --save_path ./ckpt

############################################### ROBERTA ##########################################################


TRAIN_DATA="datasets/ai_detection_500_polish.raw_data.json"

echo "=================================================="
echo "Starting RoBERTa model training..."
echo "Training data: $TRAIN_DATA"
echo "=================================================="

# 3. Define the models to train: "HF_Model_Name|Output_Directory"
ROBERTA_TRAIN_CONFIGS=(
    "openai-community/roberta-base-openai-detector|ckpt/roberta_base_ai_detector"
    "openai-community/roberta-large-openai-detector|ckpt/roberta_large_ai_detector"
)

for CONFIG in "${ROBERTA_TRAIN_CONFIGS[@]}"; do
    # Parse the configuration string
    IFS="|" read -r MODEL_NAME OUTPUT_DIR <<< "$CONFIG"

    # Create the output directory to prevent save errors
    mkdir -p "$OUTPUT_DIR"

    echo "--------------------------------------------------"
    echo "Training Model: $MODEL_NAME"
    echo "Saving to: $OUTPUT_DIR"
    echo "--------------------------------------------------"


    python other_method/train_roberta.py \
        --model_name "$MODEL_NAME" \
        --train_data_path "$TRAIN_DATA" \
        --train_data_format "ImBD" \
        --output_dir "$OUTPUT_DIR" \
        --max_sequence_length 512 \
        --learning_rate 2e-5 \
        --num_epochs 5

    echo "Successfully finished training $MODEL_NAME!"
done