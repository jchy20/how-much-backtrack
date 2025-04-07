import pandas as pd
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm

def iterate_examples(df):
    print("\nDifferent ways to iterate through a DataFrame:")
    
    print("\n1. Using iterrows():")
    for index, row in df.iterrows():
        print(f"Index: {index}, Prompt: {row['prompt'][:50]}...")
        if index >= 2:  # Just show first 3 examples
            break
    
    print("\n2. Using itertuples():")
    for row in df.itertuples():
        print(f"Index: {row.Index}, Prompt: {row.prompt[:50]}...")
        if row.Index >= 2:
            break
    
    print("\n3. Using direct column iteration:")
    for prompt in df['prompt']:
        print(f"Prompt: {prompt[:50]}...")
        if len(prompt) > 100:  # Just show first 3 examples
            break
    
    print("\n4. Using apply():")
    def process_row(row):
        print(f"Processing prompt: {row['prompt'][:50]}...")
        return len(row['prompt'])
    
    # Just show first 3 examples
    df.head(3).apply(process_row, axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                      help="Path to the checkpoint parquet file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                      help="Name of the model to use for tokenization")
    args = parser.parse_args()

    # Load the tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load the checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    df = pd.read_parquet(args.checkpoint_path)

    # Show different iteration methods
    iterate_examples(df)

    # Initialize lists to store lengths
    lengths = []
    prompts = []

    # Process each prompt using iterrows() (most common method)
    print("\nProcessing all prompts with iterrows():")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        # Tokenize the prompt
        tokens = tokenizer.encode(row['prompt'], add_special_tokens=False)
        length = len(tokens)
        lengths.append(length)
        prompts.append(row['prompt'])

    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'prompt': prompts,
        'token_length': lengths
    })

    # Calculate statistics
    print("\nToken Length Statistics:")
    print(f"Mean length: {results_df['token_length'].mean():.2f}")
    print(f"Max length: {results_df['token_length'].max()}")
    print(f"Min length: {results_df['token_length'].min()}")
    print(f"Median length: {results_df['token_length'].median():.2f}")
    print(f"95th percentile: {results_df['token_length'].quantile(0.95):.2f}")
    print(f"99th percentile: {results_df['token_length'].quantile(0.99):.2f}")

    # Save results
    output_path = args.checkpoint_path.replace('.parquet', '_token_lengths.parquet')
    results_df.to_parquet(output_path)
    print(f"\nResults saved to {output_path}")

    # Print some examples of longest prompts
    print("\nTop 5 longest prompts:")
    longest_prompts = results_df.nlargest(5, 'token_length')
    for idx, row in longest_prompts.iterrows():
        print(f"\nLength: {row['token_length']}")
        print(f"Prompt: {row['prompt'][:200]}...")  # Print first 200 chars

if __name__ == "__main__":
    main() 