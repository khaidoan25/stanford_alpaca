: "${MODEL_PATH?environment variable MODEL_PATH is unset}"
: "${OUTPUT_DIR?environment variable OUTPUT_DIR is unset}"
: "${NUM_GPU?environment variable NUM_GPU is unset}"

torchrun --standalone --nproc_per_node=$NUM_GPU train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path ./data/alpaca_5000.json \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \

