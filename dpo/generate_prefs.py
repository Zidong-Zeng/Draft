#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, random, argparse, re, asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import jsonlines
from tqdm import tqdm
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import torch
from vllm import SamplingParams, LLM

from reward import (
    check_draft_format,
    check_answer_correct,
)

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "EMPTY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:11453/v1")

aclient = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

DRAFT_PROMPT_TEMPLATE = (
    "Please write a draft answer to the following question. "
    "The draft should include your initial thoughts and reasoning steps.\n"
    "*Question*: {question}\n"
)

ROLLOUT_PROMPT_TEMPLATE = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
    "I will provide you with some prior knowledge as a draft to assist you in solving the question.\n"
    "*Question*: {question}\n*Draft*: {draft}\n"
)

# ========= 配置 =========
@dataclass
class GenConfig:
    kd: int = 6
    temp_draft: float = 0.5
    top_p_draft: float = 0.95
    max_new_tokens_draft: int = 384

    # 7B rollout 参数
    temp_rollout: float = 0.7
    top_p_rollout: float = 0.9
    max_new_tokens_rollout: int = 768

    # 运行时参数
    max_draft_tokens: int = 128      # 草稿智能截断上限（token 数）
    batch_size: int = 64             # 3B 草稿批大小（按显存调）
    concurrency_rollout: int = 128   # 7B rollout 并发上限（按服务端能力调）

    seed: int = 42
    out_path: str = "/data2/zengzidong/LLaMA-Factory/data/dpo_3Bdraft_prefs.jsonl"


def log(msg: str, path: Optional[str] = None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ========= 工具函数 =========
def backoff_sleep(try_idx: int):
    time.sleep(min(1.0 * (2 ** try_idx), 10.0))


def smart_truncate_by_tokens(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int = 128,
) -> str:
    """
    先按 token 截到 max_tokens，再“向后读到最近的逗号/句号”（中英文 , ， . 。）。
    若找不到标点，则返回纯 token 截断结果。
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text

    # 先纯 token 级截断
    rough = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

    # 在原文中定位 rough 的结尾附近，以继续向后搜标点
    anchor = rough[-30:] if len(rough) >= 30 else rough
    pos = -1
    if anchor:
        pos = text.find(anchor)
    start_idx = pos + len(anchor) if pos != -1 else len(rough)

    tail = text[start_idx:]
    m = re.search(r"[，,。\.]", tail)
    if m:
        end = start_idx + m.start() + 1
        return text[:end]
    else:
        return rough


# ========= Prompt 构造 =========
def format_chat_user(tokenizer: AutoTokenizer, content: str) -> str:
    """把单轮 user 消息转为模型标准聊天串。"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_draft_chat(tokenizer: AutoTokenizer, question: str) -> str:
    """3B 草稿聊天串"""
    prompt = DRAFT_PROMPT_TEMPLATE.format(question=question)
    return format_chat_user(tokenizer, prompt)


def build_rollout_chat(tokenizer: AutoTokenizer, question: str, draft: str) -> str:
    """7B rollout 聊天串（如改为本地 vLLM 也可直接用）"""
    prompt = ROLLOUT_PROMPT_TEMPLATE.format(question=question, draft=draft)
    return format_chat_user(tokenizer, prompt)


# ========= 批量 3B 草稿生成 =========
def gen_drafts_batch(
    llm_3b: LLM,
    tokenizer: AutoTokenizer,
    problems: List[str],
    kd: int,
    temp: float,
    top_p: float,
    max_new_tokens: int,
) -> List[List[str]]:
    """
    输入：problems = 多道题
    输出：List[List[str]]，每道题对应 kd 条草稿
    """
    prompts = [build_draft_chat(tokenizer, q) for q in problems]
    sampling_params = SamplingParams(
        n=kd,
        temperature=temp,
        top_p=top_p,
        max_tokens=max_new_tokens,
        repetition_penalty=1.05,
    )
    outs = llm_3b.generate(prompts, sampling_params)
    assert len(outs) == len(prompts), f"[3B] outputs != inputs: {len(outs)} vs {len(prompts)}"
    # 还原形状：每题 kd 个
    return [[cand.text.strip() for cand in o.outputs] for o in outs]


# ========= 并发 7B rollout（OpenAI 兼容接口）=========
async def _rollout_one_async(
    model_7b: str,
    question: str,
    draft: str,
    temp: float,
    top_p: float,
    max_new_tokens: int,
):
    """单条 draft 的异步调用。"""
    try:
        content = ROLLOUT_PROMPT_TEMPLATE.format(question=question, draft=draft)
        messages = [{"role": "user", "content": content}]
        resp = await aclient.chat.completions.create(
            model=model_7b,
            messages=messages,
            temperature=temp,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=1,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[ERROR] rollout failed: {e}"


async def rollout_7b_batch_async(
    model_7b: str,
    questions: List[str],
    drafts_list: List[List[str]],
    temp: float,
    top_p: float,
    max_new_tokens: int,
    concurrency: int = 128,
) -> List[List[str]]:
    """
    输入：
      - questions: [q1, q2, ...]
      - drafts_list: [[d11, d12, ... d1k], [d21, d22, ...], ...]
    输出：
      - rollouts_list: 与 drafts_list 形状一致
    """
    sem = asyncio.Semaphore(concurrency)

    async def _task(q, d):
        async with sem:
            return await _rollout_one_async(model_7b, q, d, temp, top_p, max_new_tokens)

    tasks = []
    for q, drafts in zip(questions, drafts_list):
        for d in drafts:
            tasks.append(_task(q, d))

    flat_outs = await asyncio.gather(*tasks)
    # 还原形状
    it = iter(flat_outs)
    grouped: List[List[str]] = []
    for drafts in drafts_list:
        grouped.append([next(it) for _ in range(len(drafts))])
    return grouped


# ========= 构造偏好对 =========
def build_pairs_for_one_problem(
    question: str,
    drafts: List[str],
    rollouts: List[str],
    reference: str,
) -> List[Dict[str, Any]]:
    """
    仅按答案正确性打分（正确=1，错误=0）。
    - 若全对或全错，返回空。
    - 否则从正确组随机采 1 个为 chosen，从错误组随机采 1 个为 rejected。
    """
    scores = [1.0 if check_answer_correct(r, reference) else 0.0 for r in rollouts]
    correct_ids  = [i for i, s in enumerate(scores) if s == 1.0]
    incorrect_ids = [i for i, s in enumerate(scores) if s == 0.0]

    if len(correct_ids) == 0 or len(incorrect_ids) == 0:
        print("no pairs (all-correct or all-wrong)")
        return []

    chosen_idx   = random.choice(correct_ids)
    rejected_idx = random.choice(incorrect_ids)

    pair = {
        "prompt": DRAFT_PROMPT_TEMPLATE.format(question=question),
        "chosen": drafts[chosen_idx],
        "rejected": drafts[rejected_idx],
        "meta": {
            "idx_chosen": chosen_idx,
            "idx_rejected": rejected_idx,
            "score_chosen": scores[chosen_idx],
            "score_rejected": scores[rejected_idx],
            "strategy": "correct_vs_incorrect_only",
        },
    }
    return [pair]


# ========= 主流程 =========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_3b", type=str,
                        default="/data2/wuzhuoyang/study-from-wrong/model/qwen2.5-3b-instruct/Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model_7b", type=str, default="qwen25-7b")
    parser.add_argument("--train_data_path", type=str,
                        default="/data2/zengzidong/Draft/dpo/test_prefs.jsonl")
    parser.add_argument("--out_path", type=str,
                        default="/data2/zengzidong/Draft/dpo/result_prefs.jsonl")
    parser.add_argument("--kd", type=int, default=6)
    parser.add_argument("--max_draft_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--concurrency_rollout", type=int, default=128)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    parser.add_argument("--max_model_len_3b", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    # 量化/设备（保留接口，当前脚本使用 vLLM，不直接用 transformers 的 load_in_8bit/4bit）
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="auto")
    args = parser.parse_args()

    random.seed(args.seed)

    # 组装运行配置
    cfg = GenConfig(
        kd=args.kd,
        max_draft_tokens=args.max_draft_tokens,
        batch_size=args.batch_size,
        concurrency_rollout=args.concurrency_rollout,
        out_path=args.out_path,
    )

    # 3B 模型用 vLLM 本地加载
    tokenizer = AutoTokenizer.from_pretrained(args.model_3b, trust_remote_code=True)
    llm_3b = LLM(
        model=args.model_3b,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len_3b,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    print("3B 模型加载成功！")

    trunc_tokenizer = tokenizer

    # 读取题库到内存（如果特别大，可自行改为流式读取）
    data: List[Tuple[str, str]] = []
    with open(args.train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 支持 {problem, solution} 键（你的数据格式）
            if "problem" in obj and "solution" in obj:
                data.append((obj["problem"], obj["solution"]))

    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    total_pairs = 0

    with jsonlines.open(cfg.out_path, mode="w") as writer:
        # 按 batch 处理
        for i in tqdm(range(0, len(data), cfg.batch_size), desc="Generating draft prefs (batched)"):
            batch = data[i:i + cfg.batch_size]
            problems = [p for p, _ in batch]
            solutions = [s for _, s in batch]

            # 1) 3B 批量采样 kd 条草稿/题
            drafts_list = gen_drafts_batch(
                llm_3b, tokenizer, problems,
                kd=cfg.kd,
                temp=cfg.temp_draft,
                top_p=cfg.top_p_draft,
                max_new_tokens=cfg.max_new_tokens_draft,
            )

            # 2) 智能截断（按 token 上限）
            drafts_list = [
                [smart_truncate_by_tokens(d, trunc_tokenizer, cfg.max_draft_tokens) for d in drafts]
                for drafts in drafts_list
            ]

            # 3) 7B 并发 rollout（OpenAI 兼容接口，服务器端会合批）
            rollouts_list = asyncio.run(rollout_7b_batch_async(
                model_7b=args.model_7b,
                questions=problems,
                drafts_list=drafts_list,
                temp=cfg.temp_rollout,
                top_p=cfg.top_p_rollout,
                max_new_tokens=cfg.max_new_tokens_rollout,
                concurrency=cfg.concurrency_rollout,
            ))

            # 4) 构造偏好对并写出
            for problem, solution, drafts, rollouts in zip(problems, solutions, drafts_list, rollouts_list):
                pairs = build_pairs_for_one_problem(problem, drafts, rollouts, solution)
                if not pairs:
                    continue
                for p in pairs:
                    writer.write(p)
                total_pairs += len(pairs)

    print(f"Done. Wrote {total_pairs} pairs to {cfg.out_path}")


if __name__ == "__main__":
    main()
