import jsonlines
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import time
import json
import re
import torch
import os
import csv
from tqdm import tqdm
import argparse

# Set OpenAI's API key and API base to use vLLM's API server.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def split_list(input_list, n):
    """
    将列表分成多个子列表，每个子列表包含 n 个元素。
    :param input_list: 原始列表
    :param n: 每个子列表的元素个数
    :return: 一个包含子列表的列表
    """
    return [input_list[i:i + n] for i in range(0, len(input_list), n)]

def process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params):
    data = []
    with jsonlines.open(json_filename) as infile:
        print("start")
        for item in infile:
            group = {}
            # 根据数据集类型处理不同字段
            if 'gsm8k' in json_filename.lower():
                # gsm8k数据集: question和answer字段，answer需要提取####后的内容
                question = f"""Solve the problem step by step and put your final answer within \\boxed{{}}. *Problems*: {item['question']}"""
                answer = item['answer'].split('####')[-1].strip()
            elif 'math500' in json_filename.lower():
                # math500数据集: problem和expected_answer字段
                question = f"""Solve the problem step by step and put your final answer within \\boxed{{}}. *Problems*: {item['problem']}"""
                answer = item['expected_answer']
            elif 'aime24' in json_filename.lower():
                # aime24和gpqa数据集: problem和answer字段
                question = f"""Solve the problem step by step and put your final answer within \\boxed{{}}. *Problems*: {item['question']}"""
                answer = item['answer']
            elif 'gpqa'or 'mmlu' in json_filename.lower():
                # gpqa数据集: Question和Correct_Choice字段
                question = f"""Solve the problem step by step and put your final option(A,B,C,D) within \\boxed{{}}. *Problems*: {item['Question']}"""
                answer = item['Correct_Choice']
            elif 'folio' in json_filename.lower():
                # folio数据集: question和answer字段
                question = f"""Solve the problem step by step and put your final option(True,False,Uncertain) within \\boxed{{}}.*Premises*: {item['premises']}*Conclusion*: {item['conclusion']}"""
                answer = item['label']
            else:
                # 默认处理方式
                question = f"""Solve the problem step by step and put your final answer within \\boxed{{}}. *Problems*: {item.get('question', item.get('problem', 'Unknown problem'))}"""
                answer = item.get('answer', item.get('expected_answer', 'Unknown answer'))
            
            group['question'] = question
            group['answer'] = answer
            data.append(group)

    texts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt['question']}], tokenize=False, add_generation_prompt=True) for prompt in data]
    split_texts = split_list(texts, batch_size)
    results = []

    for item in tqdm(split_texts):
        outputs = llm.generate(item, sampling_params)
        for output in outputs:
            result = output.outputs[0].text
            results.append(result)

    for result, item in zip(results, data):
        item['output'] = result

    with open(file_name, "w") as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)  # 将元素转换为 JSON 格式字符串
            file.write(json_line + "\n")  # 写入文件，每个 JSON 对象占一行
    print(f"数据已成功写入")

def main():
    

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, default=None, help="本地模型路径或Hugging Face Hub模型ID")
    argparser.add_argument("--json_filename", type=str, default=None, help="输入的jsonl文件路径")
    argparser.add_argument("--file_name", type=str, default=None, help="输出的jsonl文件路径")
    args = argparser.parse_args()

    model_path = args.model_path
    json_filename = args.json_filename
    file_name = args.file_name


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    sampling_params = SamplingParams(n=1, temperature=0.8, top_p=0.9, repetition_penalty=1.05, max_tokens=4096)
    llm = LLM(
        model= model_path, gpu_memory_utilization=0.9, max_model_len=4096, trust_remote_code=True, tensor_parallel_size=1
    )
    batch_size = 64
    process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params)

if __name__ == "__main__":
    main()


