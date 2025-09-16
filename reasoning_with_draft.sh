
nohup python /data2/zengzidong/Draft/test/reasoning_with_draft.py \
  --input /data2/zengzidong/Draft/test/draft_of_3B/original/gsm8k_draft_1.jsonl  \
  --output /data2/zengzidong/Draft/test/answer_of_7B/original/gsm8k-test-alltokens-1.jsonl\
  --model_path /data2/jiyifan/plm_dir/Qwen2.5-7B-Instruct \
  --gpu 2 \
  --batch_size 256 \
  --max_new_tokens 8192 \
  --max_model_len 8192 \
  > math500_draft.log 2>&1 &
