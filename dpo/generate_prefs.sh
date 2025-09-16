#!/bin/bash
# 仅 7B 走 vLLM API；3B 在本地由 transformers 加载

PYTHON_BIN="/data1/zengzidong/conda/envs/unsloth_env/bin/python"
SCRIPT_PATH="/data2/zengzidong/Draft/dpo/generate_prefs_2.py"

# 3B 本地路径（注意：这里必须是本地模型目录/权重）
MODEL_3B="/data2/wuzhuoyang/study-from-wrong/model/qwen2.5-3b-instruct/Qwen/Qwen2.5-3B-Instruct"
# 7B 用 vLLM 对外名字（非路径！）
MODEL_7B="qwen25-7b"   # 确认你的 vLLM 服务器里暴露的 name

TRAIN_DATA="/data2/zengzidong/dataset/math/train.jsonl"
OUT_DATA="/data2/zengzidong/Draft/dpo/prefs/result_prefs_0.5.jsonl"

# vLLM OpenAI 服务
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:11453/v1"

# 把本进程绑到 GPU1，避免与 vLLM 的 7B (GPU0) 冲突
export CUDA_VISIBLE_DEVICES=2

LOG_DIR="/data2/zengzidong/Draft/dpo/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_gen_prefs_draft_tem0.5.log"

# 若显存紧张，可加 --load_in_8bit 或 --load_in_4bit（二选一）
nohup $PYTHON_BIN $SCRIPT_PATH \
  --model_3b "$MODEL_3B" \
  --model_7b "$MODEL_7B" \
  --train_data_path "$TRAIN_DATA" \
  --out_path "$OUT_DATA" \
  --kd 4 \
  --max_draft_tokens 128 \
  --device_map "auto" \
  --torch_dtype "auto" \
  > "$LOG_FILE" 2>&1 &

echo "✅ gen_prefs_draft_2.py 已启动（3B 本地，7B vLLM）。日志：$LOG_FILE"
