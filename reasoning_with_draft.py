#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import jsonlines
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =========================
# 实用函数：分批
# =========================
def split_list(input_list, n):
    return [input_list[i:i + n] for i in range(0, len(input_list), n)]

# =========================
# 文本截断：按 token 上限 + 尽量对齐句末
# =========================
def truncate_text_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int,
    try_extend_to_sentence_end: bool = True,
) -> str:
    """按 token 上限截断；若可能，向后扩展到句末标点（不超过上限）"""
    if not text:
        return text

    encode = tokenizer.encode
    decode = tokenizer.decode

    tokens = encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    truncated_text = decode(truncated_tokens, skip_special_tokens=True)

    if not try_extend_to_sentence_end:
        return truncated_text

    # 尝试在原文中找到一个合理的句末标点进行小幅扩展
    # 注意：decode 后的文本与原文本在空格/正规化上可能有轻微差异，保守处理
    rest = text[len(truncated_text):]
    m = re.search(r'[\.。!?！？,，；;]', rest)
    if m:
        candidate = truncated_text + rest[:m.end()]
        if len(encode(candidate, add_special_tokens=False)) <= max_tokens:
            return candidate

    return truncated_text

# =========================
# 清洗 prompt：移除训练前后缀
# =========================
POSSIBLE_PREFIXES = [
    # 你之前常见的几种前缀都列出来做鲁棒匹配
    "Please write a draft answer to the following question. The draft should include your initial thoughts and reasoning steps. *Question*:",
    "\nYou are given a reasoning problem. Your task is not to solve it completely. Instead, draft only the first reasoning step or initial setup that guides the reasoning process.\nWrap your output between the tags:\n<|begin of draft|> and <|end of draft|>.",
    "Below is a reasoning problem. Your task is not to solve it completely. Instead, provide only a brief draft — the initial thinking steps or setup that can guide the reasoning process. Do not write the full solution or final answer.",
]
POSSIBLE_SUFFIXES = [
    "<|im_end|>\n<|im_start|>assistant\n",
]

def process_prompt_content(original_prompt: str) -> str:
    """去掉可能的前后缀，保持纯题面"""
    if not original_prompt:
        return original_prompt

    processed = original_prompt
    for prefix in POSSIBLE_PREFIXES:
        if processed.startswith(prefix):
            processed = processed[len(prefix):]
            break

    for suffix in POSSIBLE_SUFFIXES:
        if processed.endswith(suffix):
            processed = processed[:-len(suffix)]
            break

    return processed.strip()

# =========================
# Prompt 构建
# =========================
def build_prompt_math500(question: str, truncated_draft: str) -> str:
    return (
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "I will provide you with some prior knowledge as a draft to assist you in solving the question.\n"
        f"*Question*: {question}\n {truncated_draft}"
    )

def build_prompt_gpqa(question: str, truncated_draft: str) -> str:
    return (
        "Please reason step by step, and put your final option (A,B,C,D) within \\boxed{}.\n"
        "I will provide you with some prior knowledge as a draft to assist you in solving the question.\n"
        f"*Question*: {question}\n {truncated_draft}"
    )

def build_prompt_folio(question: str, truncated_draft: str) -> str:
    return (
        "Please reason step by step, and put your final option (True,False,Uncertain) within \\boxed{}.\n"
        "I will provide you with some prior knowledge as a draft to assist you in solving the question.\n"
        f"*Question*: {question}\n {truncated_draft}"
    )

PROMPT_BUILDERS = {
    "math500": build_prompt_math500,
    "gpqa": build_prompt_gpqa,
    "folio": build_prompt_folio,
    "mmlu": build_prompt_gpqa,  # mmlu 多为选择题，类似 gpqa
}

# =========================
# 数据集字段抽取
# =========================
def extract_qna_by_filename(item: dict, json_filename: str):
    """
    依据文件名识别数据集，抽取“纯题面”和“答案”，以及选用的 prompt builder key。
    该函数用于“原始数据集格式”；若是 SFT draft 文件（含 output 为草稿），请用 parse_item() 统一处理。
    """
    fname = (json_filename or "").lower()

    if "gsm8k" in fname:
        question_text = item.get("question", "")
        answer = str(item.get("answer", "")).split("####")[-1].strip()
        return question_text, answer, "math500"

    if "math500" in fname:
        return item.get("problem", ""), item.get("expected_answer", ""), "math500"

    if "aime24" in fname:
        return item.get("question", ""), item.get("answer", ""), "math500"

    if "gpqa" in fname or "mmlu" in fname:
        return item.get("Question", ""), item.get("Correct_Choice", ""), "gpqa"   # A/B/C/D

    if "folio" in fname:
        q = f"Premises: {item.get('premises','')}  Conclusion: {item.get('conclusion','')}"
        return q, item.get("label", ""), "folio"

    # 兜底
    q = item.get("question", item.get("problem", "Unknown problem"))
    a = item.get("answer", item.get("expected_answer", "Unknown answer"))
    return q, a, "math500"

def looks_like_sft_draft_file(filename: str) -> bool:
    f = (filename or "").lower()
    return any(k in f for k in ["sft_draft", "sft", "draft"])

def parse_item(item: dict, json_filename: str):
    """
    统一抽取：
    - question_text: 纯题面
    - answer: 标准答案（若有）
    - key: prompt builder key
    - draft: 草稿（若有）
    兼容两类输入：
      1) SFT draft 风格（question + answer + output 作为草稿）
      2) 原始数据集风格（字段随数据集不同）
    """
    if looks_like_sft_draft_file(json_filename):
        question_text = process_prompt_content(item.get("question", ""))
        answer = item.get("answer", "")
        draft = item.get("output", "")
        key = "math500" if not any(k in json_filename.lower() for k in ["gpqa", "folio", "mmlu"]) else \
      ("gpqa" if "gpqa" in json_filename.lower() or "mmlu" in json_filename.lower() else "folio")
        return question_text, answer, key, draft

    # 原始数据集：尝试按文件名 schema 抽取
    q, a, key = extract_qna_by_filename(item, json_filename)
    draft = item.get("output", "")  # 若本身也带有草稿则利用；没有就空
    return q, a, key, draft

# =========================
# Chat 渲染：优先用 apply_chat_template
# =========================
def render_chat(tokenizer, prompt_text: str) -> str:
    """
    将 prompt_text 转为 chat 格式（Qwen 等模型）；若不可用则直接返回文本。
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        return prompt_text

# =========================
# 主流程
# =========================
def process_data(
    json_filename: str,
    output_path: str,
    llm,
    tokenizer,
    batch_size: int,
    sampling_params: SamplingParams,
    max_draft_tokens: int,
):
    payloads = []

    with jsonlines.open(json_filename) as infile:
        for item in infile:
            question_text, answer, key, draft = parse_item(item, json_filename)
            truncated_draft = truncate_text_by_tokens(
                draft, tokenizer, max_tokens=max_draft_tokens, try_extend_to_sentence_end=True
            ) if draft else ""

            # 选择 Prompt Builder（默认用 math500）
            builder = PROMPT_BUILDERS.get(key, build_prompt_math500)
            prompt_text = builder(question_text, truncated_draft)

            payloads.append({
                "question": prompt_text,     # 注意：这是“构建后的 prompt”
                "answer": answer,            # 标准答案（若有）
                "meta": {
                    "dataset_key": key,
                    "has_draft": bool(draft),
                    "draft_tokens_limit": max_draft_tokens,
                }
            })

    # 渲染 chat，并切分批次
    rendereds = [render_chat(tokenizer, p["question"]) for p in payloads]
    batches = split_list(rendereds, batch_size)

    results = []
    for batch in tqdm(batches, desc="Generating with vLLM"):
        outputs = llm.generate(batch, sampling_params)
        for out in outputs:
            results.append(out.outputs[0].text)

    # 写回
    with open(output_path, "w", encoding="utf-8") as f:
        for gen, item in zip(results, payloads):
            record = {
                "question": item["question"],  # 已构建的 prompt
                "answer": item["answer"],
                "output": gen,
                "meta": item["meta"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ 推理完成，结果已写入：{output_path}")

# =========================
# 入口
# =========================
def main():
    parser = argparse.ArgumentParser(description="Draft-then-Reason 批量推理融合脚本")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入 JSONL（可为 sft_draft.jsonl 或原始数据集）")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--model_path", type=str, default="/data2/jiyifan/plm_dir/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="不填则与 model_path 相同")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu", type=str, default="2", help="CUDA_VISIBLE_DEVICES，例如 '2' 或 '0,1'")
    parser.add_argument("--max_draft_tokens", type=int, default=9999, help="草稿可见范围 token 上限（如需保留完整草稿可设很大，如 9999）")
    parser.add_argument("--gpu_mem_util", type=float, default=0.95)
    args = parser.parse_args()

    # 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Tokenizer
    tok_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    print("输入草稿路径：",args.input)
    print("输出答案路径：",args.output)


    # vLLM
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
    )
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )

    process_data(
        json_filename=args.input,
        output_path=args.output,
        llm=llm,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        sampling_params=sampling_params,
        max_draft_tokens=args.max_draft_tokens,
    )

if __name__ == "__main__":
    main()
