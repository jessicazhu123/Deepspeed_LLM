mkdir -p ./mistral_7B_distillation_V2_Plus
deepspeed main.py \
   --data_path ./data_input/dpo_data_v2/ \
   --model_name_or_path ./mistral_7B_distillation_V2/ \
   --data_output_path ./data_output \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 8192 \
   --gradient_checkpointing \
   --learning_rate 5e-7 \
   --weight_decay 0.1 \
   --num_train_epochs 3 \
   --num_warmup_steps 0 \
   --zero_stage 3 \
   --eval_interval 1000 \
   --eval_iters 500 \
   --deepspeed \
   --dtype bf16 \
   --output_dir ./mistral_7B_distillation_V2_Plus