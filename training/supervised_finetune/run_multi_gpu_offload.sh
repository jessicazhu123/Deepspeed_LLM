mkdir -p ./llama3_8B_sft_output_offload/
deepspeed main.py --data_path ./data_input/ --model_name_or_path ../../../../GPT_For_Personalization/llm_model/Llama-3-8b-hf/ --data_output_path ./data_output_llama3_8B \
   --max_seq_len 8192 --learning_rate 5e-6 --num_train_epochs 10 --gradient_checkpointing --zero_stage 3 --deepspeed \
   --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --checkpoint_steps 120 --output_dir ./llama3_8B_sft_output_offload \
   --offload --optimizer adam-cpu --compute_fp32_loss --num_warmup_steps 100 --dtype bf16