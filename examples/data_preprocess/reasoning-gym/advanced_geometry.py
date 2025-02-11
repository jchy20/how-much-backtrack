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


def make_prefix(dp, template_type, task_type):
    question = dp['question']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        if task_type == 'incircle_radius':
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question} Show your work in <think> </think> tags. And return only the final numeric answer in <answer> </answer> tags and round it to 3 decimal places, for example <answer> 2.176 </answer>.
Assistant: Let me solve this step by step.
<think>"""
        elif task_type == 'orthocenter':
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question} Show your work in <think> </think> tags. And return only the final numeric answer in <answer> </answer> tags and round it to 3 decimal places, for example <answer> 2.176 </answer>.
Assistant: Let me solve this step by step.
<think>"""
        elif task_type == 'angle_measure':
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question} Show your work in <think> </think> tags. And return only the final numeric answer in <answer> </answer> tags and round it to 3 decimal places, for example <answer> 2.176 </answer>.
Assistant: Let me solve this step by step.
<think>"""

    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        if task_type == 'incircle_radius':
            prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n{question}. Show your work in <think> </think> tags. And return only the final numeric answer in <answer> </answer> tags and round it to 3 decimal places, for example <answer> 2.176 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
        elif task_type == 'orthocenter':
            prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n{question}. Show your work in <think> </think> tags. And return only the final numeric answer in <answer> </answer> tags and round it to 3 decimal places, for example <answer> 2.176 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
        elif task_type == 'angle_measure':  
            prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n{question}. Show your work in <think> </think> tags. And return only the final numeric answer in <answer> </answer> tags and round it to 3 decimal places, for example <answer> 2.176 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='advanced_geometry')
    parser.add_argument('--local_dir', default='~/data/advanced_geometry')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--val_size', type=int, default=1024)
    parser.add_argument('--seed', default=42)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--task_type', type=str, default='incircle_radius')
    args = parser.parse_args()

    train_dataset = reasoning_gym.create_dataset(args.dataset_name, size=args.train_size, seed=args.seed, task_types=[args.task_type])
    val_dataset = reasoning_gym.create_dataset(args.dataset_name, size=args.val_size, seed=args.seed, task_types=[args.task_type])


    # Function to process each example into the desired format
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type, task_type=args.task_type)
            solution = example['answer']
            data = {
                "data_source": args.dataset_name,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": args.task_type,
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
