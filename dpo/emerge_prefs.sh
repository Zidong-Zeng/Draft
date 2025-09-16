nohup python /data2/zengzidong/Draft/dpo/emerge_prefs_2.py \
  --in-dir /data2/zengzidong/Draft/dpo/prefs\
  --nums 0.2,0.4,0.6,0.8,1.0 \
  --out /data2/zengzidong/LLaMA-Factory/data/draft_prefs_dpo.json\
  > /data2/zengzidong/Draft/dpo/logs/emerge_prefs.log 2>&1 &
