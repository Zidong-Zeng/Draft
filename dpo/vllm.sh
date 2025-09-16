CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model /data2/jiyifan/plm_dir/Qwen2.5-7B-Instruct \
    --served-model-name qwen25-7b \
    --max-model-len 4096 \
    --trust-remote-code  \
    --gpu-memory-utilization 0.6 \
    --port 11453 \
   > qwen2.5_7b_api_server02.log 2>&1 &
#    --chat-template "{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}\n{%- for message in messages -%}\n    {%- if message['role'] == 'user' -%}\n        {{- '<reserved_106>' + message['content'] -}}\n    {%- elif message['role'] == 'assistant' -%}\n        {{- '<reserved_107>' + message['content'] -}}\n    {%- endif -%}\n{%- endfor -%}\n\n{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}\n    {{- '<reserved_107>' -}}\n{% endif %}"
#    --distributed-executor-backend ray \
#    --pipeline-parallel-size 2
