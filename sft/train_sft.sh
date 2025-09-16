#!/bin/bash

# 设置 CUDA 设备可见性
export CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1
# 运行训练命令
# /data2/wuzhuoyang/model/qwen2.5-7b-base/Qwen/Qwen2.5-7B
# /data3/xiongqiushi/model/Meta-Llama-3-8B-Instruct
 nohup llamafactory-cli train \
  --stage sft \
  --do_train True \
  --model_name_or_path /data2/wuzhuoyang/study-from-wrong/model/qwen2.5-3b-instruct/Qwen/Qwen2.5-3B-Instruct \
  --dataset  Draft-math-qwq-0.2radio \
  --dataset_dir ./data \
  --template qwen \
  --finetuning_type lora \
  --output_dir /data2/zengzidong/LLaMA-Factory/checkpoints/sft_3B \
  --overwrite_output_dir \
  --cutoff_len 2048 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --warmup_ratio 0.1 \
  --save_steps 200 \
  --eval_steps 50 \
  --eval_strategy steps \
  --learning_rate 5.0e-5 \
  --num_train_epochs 4.0 \
  --val_size 0.1 \
  --plot_loss True \
  --bf16 \
  --report_to tensorboard \
  > /data2/zengzidong/LLaMA-Factory/checkpoints/sft_3B/output.log 2>&1 &
#  --warmup_steps 20 \

  #  --pref_beta 0.1 \
#  --pref_loss sigmoid \

  # --save_strategy steps\
  # --evaluation_strategy steps \
