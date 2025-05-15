export TOKENIZERS_PARALLELISM=false

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

DATA="${SCRIPT_DIR}/../data_shareGPT/"
NAME="shareGPT"

for eps in 4 2 1
do

OUTPUT_DIR="${SCRIPT_DIR}/../results/fine_tune_eps_${eps}"
CHECKPOINT_FOLDER="${SCRIPT_DIR}/../results/fine_tune_eps_${eps}"

# FINE-TUNING
python ${SCRIPT_DIR}/fine-tune-dp.py \
    --data_dir $DATA \
    --data_name $NAME \
    --output_dir $OUTPUT_DIR \
    --model_name gpt2 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy no \
    --save_strategy no \
    --log_level info \
    --per_device_eval_batch_size 64 \
    --eval_accumulation_steps 1 \
    --seed 42 \
    --target_epsilon $eps \
    --per_sample_max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 100 \
    --logging_steps 500 \
    --max_grad_norm 0 \
    --sequence_len 1024 \
    --learning_rate 0.0001 \
    --lr_scheduler_type constant \
    --dataloader_num_workers 2 \
    --disable_tqdm True \
    --load_best_model_at_end False \
    --save_safetensors False

# GENERATION
python ${SCRIPT_DIR}/generate-text.py \
    --model_type gpt2 \
    --model_name_or_path $CHECKPOINT_FOLDER \
    --output_dir $OUTPUT_DIR \
    --length 1024 \
    --total_sequences 600 \
    --do_sample \
    --batch_size 8 \

done