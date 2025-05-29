# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the ReasoningGym dataset to parquet format with a 80/20 train/val split.
"""

import re
import os
import datasets
import argparse
import reasoning_gym

def make_prefix(dp, template_type):
    question = dp['question']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""

    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n{question}Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    
    elif template_type == 'baseline':
        """This works for any base model"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n{question}Return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/countdown')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--val_size', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=None)
    parser.add_argument('--sft_size', type=int, default=None)
    parser.add_argument('--baseline_size', type=int, default=None)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--min_numbers', type=int, default=4)
    parser.add_argument('--max_numbers', type=int, default=6)
    parser.add_argument('--min_value', type=int, default=1)
    parser.add_argument('--max_value', type=int, default=100)
    parser.add_argument('--min_target', type=int, default=100)
    parser.add_argument('--max_target', type=int, default=999)
    parser.add_argument('--operators', type=lambda s: tuple(s.split(',')), default=('+', '-', '*', '/'))
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--template_type', type=str, default='base')
    args = parser.parse_args()

    if args.train_size is not None:
        train_dataset = reasoning_gym.create_dataset('countdown', 
                                                     size=args.train_size, 
                                                     seed=args.seed + 80000, 
                                                     min_numbers=args.min_numbers, 
                                                     max_numbers=args.max_numbers, 
                                                     min_value=args.min_value, 
                                                     max_value=args.max_value, 
                                                     min_target=args.min_target, 
                                                     max_target=args.max_target, 
                                                     operators=args.operators, 
                                                     shuffle=args.shuffle)
    if args.val_size is not None:
        val_dataset = reasoning_gym.create_dataset('countdown', size=args.val_size, 
                                                   seed=args.seed + 20, 
                                                   min_numbers=args.min_numbers, 
                                                   max_numbers=args.max_numbers, 
                                                   min_value=args.min_value, 
                                                   max_value=args.max_value, 
                                                   min_target=args.min_target, 
                                                   max_target=args.max_target, 
                                                   operators=args.operators, 
                                                   shuffle=args.shuffle)
    if args.test_size is not None:
        test_dataset = reasoning_gym.create_dataset('countdown', 
                                                    size=args.test_size, 
                                                    seed=args.seed + 250000, 
                                                    min_numbers=args.min_numbers, 
                                                    max_numbers=args.max_numbers, 
                                                    min_value=args.min_value, 
                                                    max_value=args.max_value, 
                                                    min_target=args.min_target, 
                                                    max_target=args.max_target, 
                                                    operators=args.operators, 
                                                    shuffle=args.shuffle)
    if args.baseline_size is not None:
        baseline_dataset = reasoning_gym.create_dataset('countdown', 
                                                        size=args.baseline_size, 
                                                        seed=args.seed + 40, 
                                                        min_numbers=args.min_numbers, 
                                                        max_numbers=args.max_numbers, 
                                                        min_value=args.min_value, 
                                                        max_value=args.max_value, 
                                                        min_target=args.min_target, 
                                                        max_target=args.max_target, 
                                                        operators=args.operators, 
                                                        shuffle=args.shuffle)
    if args.sft_size is not None:
        sft_dataset = reasoning_gym.create_dataset('countdown', 
                                                    size=args.sft_size, 
                                                    seed=args.seed + 60, 
                                                    min_numbers=args.min_numbers, 
                                                    max_numbers=args.max_numbers, 
                                                    min_value=args.min_value, 
                                                    max_value=args.max_value, 
                                                    min_target=args.min_target, 
                                                    max_target=args.max_target, 
                                                    operators=args.operators, 
                                                    shuffle=args.shuffle)
    # Function to process each example into the desired format
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = example['answer']
            
            data = {
                "data_source": "countdown",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "deductive, inductive",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                    "one_backtrack": example['metadata']['one_backtrack'],
                    "two_backtrack": example['metadata']['two_backtrack'],
                    "three_backtrack": example['metadata']['three_backtrack'],
                    "optimal_trajectory": example['metadata']['optimal_trajectory']
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    # Apply processing to both splits
    if args.train_size is not None:
        mapped_train = [make_map_fn('train')(example, idx) for idx, example in enumerate(train_dataset)]
    if args.val_size is not None:
        mapped_val = [make_map_fn('val')(example, idx) for idx, example in enumerate(val_dataset)]
    if args.test_size is not None:
        mapped_test = [make_map_fn('test')(example, idx) for idx, example in enumerate(test_dataset)]
    if args.baseline_size is not None:
        mapped_baseline = [make_map_fn('baseline')(example, idx) for idx, example in enumerate(baseline_dataset)]
    if args.sft_size is not None:
        mapped_sft = [make_map_fn('sft')(example, idx) for idx, example in enumerate(sft_dataset)]
        

    # Save each split to parquet files
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    from datasets import Dataset
    if args.train_size is not None:
        train_dataset = Dataset.from_list(mapped_train)
        train_dataset.to_parquet(os.path.join(local_dir, f'{args.template_type}_train.parquet'))
    if args.val_size is not None:
        val_dataset = Dataset.from_list(mapped_val)
        val_dataset.to_parquet(os.path.join(local_dir, f'{args.template_type}_val.parquet'))
    if args.test_size is not None:
        test_dataset = Dataset.from_list(mapped_test)
        test_dataset.to_parquet(os.path.join(local_dir, f'{args.template_type}_test.parquet'))
    if args.baseline_size is not None:
        baseline_dataset = Dataset.from_list(mapped_baseline)
        baseline_dataset.to_parquet(os.path.join(local_dir, f'{args.template_type}_baseline.parquet'))
    if args.sft_size is not None:
        sft_dataset = Dataset.from_list(mapped_sft)
        sft_dataset.to_parquet(os.path.join(local_dir, f'{args.template_type}_sft_optim.parquet'))
