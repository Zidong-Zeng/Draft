import re
import numpy as np
import math
from typing import List

# ========== 工具函数 ==========
def extract_draft(text: str) -> str:
    """提取draft内容"""
    try:
        start = text.index("<|begin of draft|>") + len("<|begin of draft|>")
        end = text.index("<|end of draft|>")
        return text[start:end].strip()
    except ValueError:
        return ""

def extract_numerical_answer(text: str) -> float:
    """从文本提取数值答案"""
    try:
        # 优先从\boxed{}提取
        boxed_match = re.search(r'\\boxed{(.*?)}', text)
        if boxed_match:
            answer_text = boxed_match.group(1)
        else:
            # 否则尝试提取最后一个数字
            answer_text = text.split('\n')[-1]
        
        # 匹配第一个数值
        number_match = re.search(r'-?\d+\.?\d*', answer_text.replace(",", ""))
        return float(number_match.group()) if number_match else 0.0
    except:
        return 0.0

def extract_xml_answer(text: str) -> str:
    """提取XML格式的答案"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# ========== Draft格式检查函数 ==========
def check_draft_format(text: str) -> bool:
    """检查数学推理draft是否包含正确的标签格式
    Args:
        text: 待检查的draft文本
    Returns:
        bool: 是否格式正确（包含<|begin of draft|>和<|end of draft|>标签）
    """
    try:
        has_begin = "<|begin of draft|>" in text
        has_end = "<|end of draft|>" in text
        correct_order = text.index("<|begin of draft|>") < text.index("<|end of draft|>")
        has_content = len(extract_draft(text).strip()) > 0
        return has_begin and has_end and correct_order and has_content
    except ValueError:
        return False

def check_answer_correct(rollout: str, reference: str) -> bool:
    """检查数学答案是否正确（绝对误差<1e-6）
    Args:
        rollout: 7B模型的完整推理输出
        reference: 标准数学答案
    Returns:
        bool: 数值答案是否正确
    """
    try:
        generated_answer = extract_numerical_answer(rollout)
        reference_answer = extract_numerical_answer(reference)
        return abs(generated_answer - reference_answer) < 1e-6
    except:
        return False

# ========== 格式奖励函数 ==========
def count_xml(text) -> float:
    """计算XML格式得分 - 适用于数学推理任务"""
    count = 0.0
    
    # 检查必要的标签和换行符
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("<answer>\n") == 1:
        count += 0.125
    if text.count("\n</answer>") == 1:
        count += 0.125
    
    # 检查是否包含数学相关的方括号格式
    import re
    if re.search(r'\[(步骤|计算|答案|结果|数值)\]', text):  # 搜索数学推理相关的方括号内容
        count += 0.125  # 找到一个符合条件的方括号内容就加分
    
    if text.count("<eoa>") == 1:
        count += 0.125
        penalty = len(text.split("<eoa>\n</answer>")[-1]) * 0.01
        count -= min(penalty, 0.375)
    
    # 检查整体格式
    try:
        parts = text.split("<reasoning>\n")
        reasoning_content = parts[1].split("\n</reasoning>\n")[0]
        answer_parts = text.split("<answer>\n")
        answer_content = answer_parts[1].split("\n</answer>")[0]
        
        # 确认answer部分包含方括号内容和eoa标记
        if reasoning_content and answer_content and re.search(r'\[[^\]]+\]', answer_content) and "<eoa>" in answer_content:
            count += 0.25
    except:
        pass
    
    return count

def format_reward_func(completions, **kwargs) -> list[float]:
    """数学推理格式奖励函数 - 检查推理过程的格式规范性"""
    contents = [completion[0]["content"] for completion in completions]
    scores = [count_xml(c) for c in contents]
    
    # 打印每个内容的格式分数
    for i, score in enumerate(scores):
        print(f"内容 {i+1} 推理格式分数: {score:.3f}")

    return scores

# 向后兼容性别名
xmlcount_reward_func = format_reward_func

# ========== Draft综合奖励函数 ==========
def calculate_combined_reward(prompt: str, draft: str, rollout: str, 
                            reference: str, weights: dict = None) -> float:
    """计算数学推理任务的综合奖励分数
    Args:
        prompt: 原始数学问题
        draft: 3B模型生成的推理起始draft
        rollout: 7B模型基于draft生成的完整数学推理
        reference: 标准数学答案
        weights: 奖励权重字典 {'format': 格式权重, 'answer': 答案权重}
    Returns:
        float: 综合奖励分数 (0.0-1.0)
    """
    if weights is None:
        weights = {'format': 0.2, 'answer': 0.8}
    
    # 1. 格式奖励 (20%)：检查draft是否包含正确的标签格式
    format_score = 1.0 if check_draft_format(draft) else 0.0
    
    # 2. 数学答案正确性 (80%)：检查7B模型输出的数值答案是否与标准答案匹配
    answer_score = 1.0 if check_answer_correct(rollout, reference) else 0.0
    
    # 3. 计算加权总分
    total_reward = (
        weights['format'] * format_score +  # 20% - Draft格式奖励
        weights['answer'] * answer_score    # 80% - 数学答案奖励
    )
    
    return total_reward

# ========== 通用评分函数 ==========
def scoring_function(prompts, completions, answer, **kwargs) -> list[float]:
    """数学推理任务的简化评分函数，主要用于draft训练"""
    print("[DEBUG-REWARD] ===== 进入数学推理评分函数 =====")
    print(f"[DEBUG-REWARD] 收到 {len(prompts)} 个数学问题和 {len(completions)} 个回复")
    
    responses = []
    for completion in completions:
        responses.append(completion[0]['content'])
    
    # 计算所有数学答案评分
    all_scores = []
    
    for idx, response in enumerate(responses):
        print(f"\n[DEBUG-REWARD] 评估第 {idx+1} 个数学推理回复...")
        current_answer = answer[idx] if idx < len(answer) else answer[0]
        
        # 基于数学答案正确性的评分
        try:
            generated_answer = extract_numerical_answer(response)
            reference_answer = extract_numerical_answer(current_answer)
            
            if abs(generated_answer - reference_answer) < 1e-6:
                score = 5.0  # 数学答案完全正确
            else:
                score = 0.0  # 数学答案不正确
        except:
            score = 0.0
        
        all_scores.append(score)
        print(f"[DEBUG-REWARD] 数学答案评分: {score:.3f}")
    
    print(f"\n所有数学推理评分: {[round(s, 3) for s in all_scores]}")
    return all_scores

# ========== Draft专用奖励函数 ==========
def draft_reward_function(prompts: List[str], drafts: List[str], 
                         rollouts: List[str], references: List[str],
                         weights: dict = None) -> List[float]:
    """数学推理Draft训练专用的奖励函数
    Args:
        prompts: 原始数学问题列表
        drafts: 3B模型生成的数学推理draft列表
        rollouts: 7B模型生成的完整数学推理列表
        references: 标准数学答案列表
        weights: 奖励权重字典 {'format': 格式权重, 'answer': 答案权重}
    Returns:
        List[float]: 每个数学推理draft的奖励分数列表
    """
    if weights is None:
        weights = {'format': 0.2, 'answer': 0.8}
    
    rewards = []
    for prompt, draft, rollout, reference in zip(prompts, drafts, rollouts, references):
        reward = calculate_combined_reward(prompt, draft, rollout, reference, weights)
        rewards.append(reward)
    
    return rewards

# ========== GRPO优势计算器 ==========
class GRPOAdvantageCalculator:
    """实现组内相对优势计算"""
    def __init__(self, beta: float = 0.2, epsilon: float = 0.2):
        self.beta = beta
        self.epsilon = epsilon

    def __call__(self, group_rewards: List[float]) -> List[float]:
        rewards = np.array(group_rewards)
        # 使用均值作为基线，而不是中位数
        baseline = np.mean(rewards)  # 修改：使用均值替代中位数
        centered = rewards - baseline
        std = np.std(centered) + 1e-6  # 添加小量防止除零
        
        # GRPO核心公式
        raw_advantages = centered / std
        clipped_advantages = np.clip(raw_advantages, -self.epsilon, self.epsilon)
        return (self.beta * clipped_advantages).tolist()

# ========== 用于兼容的虚拟函数 ==========
def qwen72b_score_function(prompts, completions, answer, **kwargs) -> list[float]:
    """兼容函数，暂时返回基础评分"""
    print("[DEBUG-REWARD] 调用qwen72b_score_function（兼容模式）")
    return scoring_function(prompts, completions, answer, **kwargs) 
