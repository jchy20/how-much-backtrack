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
Preprocess the Bespoke-Stratos17k dataset to parquet format with a 80/20 train/val split.
"""

import re
import os
import datasets
import argparse

from verl.utils.hdfs_io import copy, makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/bs17k')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    data_source = 'bespokelabs/Bespoke-Stratos-17k'

    # Load the dataset (assumes the dataset only has a train split)
    dataset = datasets.load_dataset(data_source)
    original_dataset = dataset['train']

    # Split 80% train, 20% validation (set a seed for reproducibility)
    split_datasets = original_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['test']

    # Function to process each example into the desired format
    def make_map_fn(split):
        def process_fn(example, idx):
            conversations = example.pop('conversations')
            question = conversations[0]['value']
            answer = conversations[1]['value']
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "answer": [{
                    "role": "assistant",
                    "content": answer,
                }]
            }
            return data
        return process_fn

    # Apply processing to both splits
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True)

    # Save each split to parquet files
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(local_dir, 'val.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
