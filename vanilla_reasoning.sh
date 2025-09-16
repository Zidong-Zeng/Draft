export CUDA_VISIBLE_DEVICES=1

nohup python /data2/zengzidong/Draft/test/vanilla_reasoning.py \
    --model_path /data2/jiyifan/plm_dir/Qwen2.5-7B-Instruct \
    --json_filename /data2/zengzidong/dataset/MMLU/science_merged.jsonl\
    --file_name /data2/zengzidong/Draft/answer_of_7B/vanilla/mmlu_big_evaluate.jsonl\
    > mmlu_big_vanilla.log 2>&1 &
