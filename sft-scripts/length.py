import pandas as pd
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import os
from pathlib import Path

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
parquet_paths = ['advanced_geometry_checkpoint_20000.parquet', 'arc_1d.parquet', 'sudoku_checkpoint_30000.parquet', 'color_cube_rotation.parquet', 'countdown.parquet', 'list_functions.parquet', 'self_reference.parquet', 'zebra_puzzles.parquet']
parent_path = '/home/users/hc387/data/sft_data/Qwen3b-instruct/incorrect'

def check(tokenizer, parquet_path):
    df = pd.read_parquet(parquet_path)
    df['combined_text'] = df['prompt'] + df['completion']

    # Initialize list to store lengths
    lengths = []

    # Process each combined text
    print("Tokenizing combined texts...")
    for text in tqdm(df['combined_text'], desc="Processing texts"):
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        length = len(tokens)
        lengths.append(length)

    # Calculate statistics
    print(f"\nToken Length Statistics for {parquet_path.stem}:")
    print(f"Mean length: {sum(lengths) / len(lengths):.2f}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Median length: {sorted(lengths)[len(lengths)//2]:.2f}")
    print('----' * 30)
    print('----' * 30)

for parquet_path in parquet_paths:
    path = Path(os.path.join(parent_path, parquet_path))
    check(tokenizer=tokenizer, parquet_path=path)


    
