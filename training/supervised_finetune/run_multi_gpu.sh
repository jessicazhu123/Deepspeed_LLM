mkdir -p ./mistral_sft_output/
deepspeed main.py --data_path ./data_input/ --model_name_or_path ../../../../GPT_For_Personalization/llm_model/Mistral-7B-v0.1/ --data_output_path ./data_output_mistral \
   --max_seq_len 8192 --learning_rate 1e-5 --num_train_epochs 10 --gradient_checkpointing --zero_stage 3 --deepspeed \
   --per_device_train_batch_size 5 --per_device_eval_batch_size 5 --checkpoint_steps 1800 --output_dir ./mistral_sft_output \
   --compute_fp32_loss --num_warmup_steps 100 --dtype bf16