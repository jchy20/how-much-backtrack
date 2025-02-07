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
    target = dp['target']
    numbers = dp['nums']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='leg_counting')
    parser.add_argument('--local_dir', default='~/data/leg_counting')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=10)
    parser.add_argument('--val_size', type=int, default=5)
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()

    train_dataset = reasoning_gym.create_dataset(args.dataset_name, size=args.train_size, seed=args.seed)
    val_dataset = reasoning_gym.create_dataset(args.dataset_name, size=args.val_size, seed=args.seed)


    # Function to process each example into the desired format
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example['question']
            solution = example['answer']
            data = {
                "data_source": args.dataset_name,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": args.dataset_name,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    # Apply processing to both splits
    mapped_train = [make_map_fn('train')(example, idx) for idx, example in enumerate(train_dataset)]
    mapped_val = [make_map_fn('val')(example, idx) for idx, example in enumerate(val_dataset)]

    # Save each split to parquet files
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    from datasets import Dataset
    hf_train_dataset = Dataset.from_list(mapped_train)
    hf_val_dataset = Dataset.from_list(mapped_val)
    hf_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    hf_val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))
