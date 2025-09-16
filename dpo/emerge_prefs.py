#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Iterable

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"[JSONL解析失败] {path} 第{ln}行: {e}") from e

def to_training_example(rec: Dict[str, Any]) -> Dict[str, str]:
    # 必要字段检查
    for k in ("prompt", "chosen", "rejected"):
        if k not in rec:
            raise KeyError(f"记录缺少必要字段: '{k}'。原记录: {list(rec.keys())}")
    return {
        "instruction": rec["prompt"],
        "chosen": rec["chosen"],
        "rejected": rec["rejected"],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, default=".", help="输入文件所在目录")
    ap.add_argument("--nums", type=str, default="", 
                    help="以逗号分隔的{num}列表，如: 0.1,0.5,2,3.25；留空表示用--glob")
    ap.add_argument("--glob", type=str, default="result_prefs_*.jsonl",
                    help="不指定--nums时使用的通配模式")
    ap.add_argument("--out", type=str, required=True,
                    help="输出JSON路径（将写JSON数组）")
    ap.add_argument("--skip-missing", action="store_true",
                    help="当指定的文件不存在时跳过而不报错")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)

    # 解析输入文件列表
    files: List[Path] = []
    if args.nums.strip():
        for s in args.nums.split(","):
            num = s.strip()
            fname = f"result_prefs_{num}.jsonl"
            p = in_dir / fname
            if p.exists():
                files.append(p)
            else:
                if args.skip-missing:
                    print(f"[WARN] 文件不存在，跳过: {p}")
                else:
                    raise FileNotFoundError(f"未找到输入文件: {p}")
    else:
        files = sorted(in_dir.glob(args.glob))
        if not files:
            raise FileNotFoundError(f"通配未匹配到任何文件: {in_dir / args.glob}")

    merged: List[Dict[str, str]] = []
    total_in, total_ok = 0, 0

    for fp in files:
        count_before = len(merged)
        for rec in read_jsonl(fp):
            total_in += 1
            try:
                merged.append(to_training_example(rec))
                total_ok += 1
            except Exception as e:
                # 单条坏记录不影响其他数据
                print(f"[WARN] 归一化失败，文件 {fp.name} 第{total_in}条: {e}")

        added = len(merged) - count_before
        print(f"[INFO] 读取 {fp.name}: 新增 {added} 条（累计 {len(merged)}）")

    # 确保输出目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 写 JSON 数组（UTF-8，无转义中文）
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 输入记录总数: {total_in}，成功规范化: {total_ok}。")
    print(f"[OUT ] 写出: {out_path} （{len(merged)} 条）")

if __name__ == "__main__":
    main()
