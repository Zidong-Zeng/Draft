#!/usr/bin/env python3
# coding: utf-8

import os, time, json, random
from datetime import datetime
import torch
import numpy as np
import wandb
import jsonlines
from openai import OpenAI

import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from peft import get_peft_model
from trl import GRPOConfig, GRPOTrainer

# Register GRPO support in Unsloth
PatchFastRL("GRPO", FastLanguageModel)

# Reward utilities
from reward import (
    check_draft_format,
    check_answer_correct,
    calculate_combined_reward,
    GRPOAdvantageCalculator,
)


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:11452/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def my_predict(client, prompt1):
    try:
        result = []
        chat_response = client.chat.completions.create(
            model="/data2/jiyifan/plm_dir/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "user", "content": prompt1},
            ],
            n=1,
            max_tokens=1024, # 设置最大生成token数
        )
        result = chat_response.choices[0].message.content
    except Exception as e:
        # 如果出现异常，打印错误信息，但继续执行后续代码
        print("发生异常:", e)
        result = "有异常出现"
        # formatted_dict = {'text_generation_text': [result]}
        # result = [formatted_dict]
        # 可以添加其他处理方式，例如记录日志、继续执行其他代码等
    return result


# ========== Config ==========
class TrainingConfig:
    MODEL_3B_PATH = "/data2/wuzhuoyang/study-from-wrong/model/qwen2.5-3b-instruct/Qwen/Qwen2.5-3B-Instruct"
    MODEL_7B_PATH = "/data2/jiyifan/plm_dir/Qwen2.5-7B-Instruct"
    LORA_RANK = 32

    NUM_DRAFTS = 4
    PER_DEVICE_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_SEQ_LENGTH = 4096

    GRPO_BETA = 0.2
    GRPO_EPSILON = 0.25
    REWARD_WEIGHTS = {"format": 0.2, "answer": 0.8}

    DRAFT_PROMPT_TEMPLATE = (
        "You are given a reasoning problem. Your task is not to solve it completely. "
        "Instead, draft only the first reasoning step or initial setup that guides the reasoning process."
        "Wrap your output between the tags:\n<|begin of draft|> and <|end of draft|>.\n{question}\n"
    )
    ROLLOUT_PROMPT_TEMPLATE = (
        "Please reason step by step, and put your final answer within \\boxed{{}}.I will provide you with some prior knowledge as a draft to assist you in solving the question.\n"
        "*Question*: {question}\n*Draft*: {draft}"
    )


# ========== Helpers ==========
experiment_name = f"qwen3B_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
log_file = os.path.join("logs", f"{experiment_name}.log")


def log_print(msg):
    print(msg)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def load_and_split_jsonl(path, dev_ratio=0.1, seed=42):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    random.seed(seed)
    random.shuffle(data)
    idx = int(len(data) * (1 - dev_ratio))

    formatted_data = []
    for item in data:
        formatted_item = {
            "draft_prompt": TrainingConfig.DRAFT_PROMPT_TEMPLATE.format(
                question=item["problem"]
            ),
            "original_prompt": item["problem"],
            "solution": item["solution"],
        }
        formatted_data.append(formatted_item)

    return Dataset.from_list(formatted_data[:idx]), Dataset.from_list(
        formatted_data[idx:]
    )
    # return Dataset.from_list(data[:idx]), Dataset.from_list(data[idx:])


# ========== Custom Trainer ==========


def combined_reward_func(completions, **kwargs):
    print("😂 kwargs keys:", kwargs.keys())
    # 获取原始问题和参考答案
    original_prompts = kwargs.get("original_prompt")
    solutions = kwargs.get("solution")  # 注意这里使用 "solution" 而不是 "answers"
    draft_prompt = kwargs.get("draft_prompt")

    if original_prompts is None or solutions is None:
        raise ValueError("Missing required inputs: original_prompts or solutions.")

    rewards = []
    for i, completion in enumerate(completions):
        # 计算当前在组中的位置
        group_index = i % TrainingConfig.NUM_DRAFTS
        is_first_in_group = group_index == 0

        if is_first_in_group:
            log_print(
                f"\n--- 样本 {i//TrainingConfig.NUM_DRAFTS} 的第 {group_index} 个草稿 ---"
            )
            log_print(f"初步生成的草稿内容: {completion}")
        else:
            # 非第一个样本，只显示简单信息
            log_print(
                f"\n--- 样本 {i//TrainingConfig.NUM_DRAFTS} 的第 {group_index} 个草稿 (略) ---"
            )
            log_print(f"初步生成的草稿内容: [内容略，仅显示第一个草稿]")

        # 1) 格式奖励
        fmt = 0.5 if check_draft_format(completion) else 0.0

        if is_first_in_group:
            log_print(f"✅ Format OK: {fmt > 0}")

        # 2) rollout + 准确率
        original_question = original_prompts[i // TrainingConfig.NUM_DRAFTS]
        prompt = TrainingConfig.ROLLOUT_PROMPT_TEMPLATE.format(
            question=original_question, draft=completion
        )

        if is_first_in_group:
            log_print(f"利用draft生成最终答案的prompt:\n {prompt}")

        # out = model_7b.generate([prompt], sampling_params)
        # full = out[0].outputs[0].text
        full = my_predict(client, prompt)

        if is_first_in_group:
            log_print(f"利用draft生成的最终答案: {full}")

        solution = solutions[i // TrainingConfig.NUM_DRAFTS]
        corr = 1.0 if check_answer_correct(full, solution) else 0.0

        if is_first_in_group:
            print(f"✅ ACC OK: {corr > 0}")

        rewards.append(
            TrainingConfig.REWARD_WEIGHTS["format"] * fmt
            + TrainingConfig.REWARD_WEIGHTS["answer"] * corr
        )

    # 优势计算
    # return GRPOAdvantageCalculator(
    #     beta=TrainingConfig.GRPO_BETA, epsilon=TrainingConfig.GRPO_EPSILON
    # )(rewards)
    return rewards

# ========== Main ==========
def main():
    # Logging & WandB
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logfile = f"logs/grpo_{datetime.now():%Y%m%d_%H%M%S}.log"
    os.makedirs("logs", exist_ok=True)
    log_print("Starting experiment")
    wandb.init(project="GRPO_draft", name="qwen3B", mode="offline")

    # Load models
    model_3b, tokenizer = FastLanguageModel.from_pretrained(
        TrainingConfig.MODEL_3B_PATH,
        max_seq_length=TrainingConfig.MAX_SEQ_LENGTH,
        fast_inference=True,
        max_lora_rank=TrainingConfig.LORA_RANK,
        gpu_memory_utilization=0.4,
        trust_remote_code=False,
        local_files_only=True,
        use_cache=False,
    )

    log_print("✅ FastLanguageModel loaded (base)")

    # 2) 给基础模型打 LoRA 补丁
    model_3b = FastLanguageModel.get_peft_model(
        model_3b,
        r=TrainingConfig.LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=TrainingConfig.LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    log_print("✅ FastLanguageModel LoRA setup complete")

    global model_7b, sampling_params

    # model_7b = LLM(
    #     model=TrainingConfig.MODEL_7B_PATH,
    #     gpu_memory_utilization=0.55,
    #     max_model_len=4096,
    #     trust_remote_code=False,
    # )
    # sampling_params = SamplingParams(n=1, temperature=0.7, top_p=0.9, max_tokens=4096)

    # Load & split data
    train_ds, dev_ds = load_and_split_jsonl(
        "/data2/zengzidong/dataset/math/train.jsonl", dev_ratio=0.1
    )
    # train_ds = train_ds.map(lambda x: {"prompt": x["problem"], "answer": x["solution"]})
    # dev_ds = dev_ds.map(lambda x: {"prompt": x["problem"], "answer": x["solution"]})
    train_ds = train_ds.map(
        lambda x: {
            "prompt": x["draft_prompt"],
            "solution": x["solution"],
            "original_prompt": x["original_prompt"],
        }
    )
    dev_ds = dev_ds.map(
        lambda x: {
            "prompt": x["draft_prompt"],
            "solution": x["solution"],
            "original_prompt": x["original_prompt"],
        }
    )
    log_print(f"Train size: {len(train_ds)}, Dev size: {len(dev_ds)}")

    # GRPO training args
    args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        per_device_train_batch_size=TrainingConfig.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=TrainingConfig.GRADIENT_ACCUMULATION_STEPS,
        num_generations=TrainingConfig.NUM_DRAFTS,
        num_train_epochs=3,
        save_strategy="steps",
        save_steps=50,  # 每 50 步保存一次
        eval_strategy="steps",
        eval_steps=50,  # 每 50 步评估一次
        logging_strategy="steps",
        logging_steps=1,
        logging_dir="./outputs/logs",
        max_completion_length=1024,
        report_to="tensorboard",
        output_dir="outputs",
        do_train=True,
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model_3b,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        prompt_key="prompt",
        reference_key="solution",
        reward_funcs=[combined_reward_func],
        reward_func_kwargs={
            "original_prompts": train_ds["original_prompt"],
            "solution": train_ds["solution"],
        },
    )

    # Train
    trainer.train()
    wandb.finish()
    log_print("Experiment done")


if __name__ == "__main__":
    main()
