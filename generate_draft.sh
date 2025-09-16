export CUDA_VISIBLE_DEVICES=3

nohup python /data2/zengzidong/Draft/test/generate_draft.py \
  --model_path /data2/zengzidong/LLaMA-Factory/checkpoints/qwen2_5vl-7b/lora/dpo/emerged-750-3b-dpo \
  --json_filename /data2/zengzidong/Draft/data_set/math500/test.jsonl \
  --file_name /data2/zengzidong/Draft/test/draft_of_3B/dpo/math500_draft_1.jsonl \
  --batch_size 128 \
  --max_draft_tokens 128 \
  > math500_draft.log 2>&1 &
