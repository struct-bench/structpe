SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

cls_batch_size=10
num_train_epochs=10
max_seq_length=4096
model="allenai/longformer-base-4096"
min_token_threshold=50

seed=0

#res_folder="shareGPT_vanilla"

## calculate acc 
for res_folder in 'shareGPT_vanilla' 'shareGPT_var_keep_token' 'shareGPT_reformat' 'shareGPT_extract_vanilla' 'shareGPT_var_keep_token_reformat'
do

for iter in '00' '02' '04' '06' '08' '10'
do

for func_type in "intent" "topic"
do
path="${SCRIPT_DIR}/../../results/${res_folder}/synthetic_text"

train_file="${path}/0000000${iter}_labeled.csv"
test_file="${SCRIPT_DIR}/../../data_shareGPT/shareGPT_test.csv"

echo $train_file
if [ -e "$train_file" ]; then
    echo "$train_file does exist."
    output_dir="${path}/${func_type}_classification/${iter}/"
    if [ -e "${output_dir}all_results.json" ]; then
        echo "-- SKIP running classification"
    else
        echo "-- RUN running classification"
        python ${SCRIPT_DIR}/shareGPT_classification.py \
            --report_to none  --min_token_threshold ${min_token_threshold} \
            --model_name_or_path  ${model} \
            --output_dir ${output_dir} \
            --train_file ${train_file} \
            --validation_file ${train_file} \
            --test_file ${test_file} \
            --do_train --do_predict --max_seq_length ${max_seq_length} --per_device_train_batch_size ${cls_batch_size} --per_device_eval_batch_size ${cls_batch_size} \
            --learning_rate 3e-5 --num_train_epochs ${num_train_epochs} \
            --overwrite_output_dir --overwrite_cache \
            --save_strategy no --save_total_limit 1 --load_best_model_at_end \
            --logging_strategy epoch \
            --seed ${seed} \
            --metric_for_best_model accuracy_all --greater_is_better True \
            --evaluation_strategy no --label_column_name ${func_type}
    fi
else
    echo "$train_file does not exist."
fi
done
done
done