import pandas as pd
import argparse

def split_backtrack_data(input_file):
    # Read the input parquet file
    df = pd.read_parquet(input_file)
    
    # Print the structure of the first row to debug
    print("First row structure:")
    print(df.iloc[0])
    
    # Create separate dataframes for each backtrack level
    optimal_df = pd.DataFrame({
        'prompt': df['prompt'].apply(lambda x: x[0]['content']),
        'completion': df['reward_model'].apply(lambda x: x['optimal_trajectory'])
    })
    
    one_backtrack_df = pd.DataFrame({
        'prompt': df['prompt'].apply(lambda x: x[0]['content']),
        'completion': df['reward_model'].apply(lambda x: x['one_backtrack'])
    })
    
    two_backtrack_df = pd.DataFrame({
        'prompt': df['prompt'].apply(lambda x: x[0]['content']),
        'completion': df['reward_model'].apply(lambda x: x['two_backtrack'])
    })
    
    three_backtrack_df = pd.DataFrame({
        'prompt': df['prompt'].apply(lambda x: x[0]['content']),
        'completion': df['reward_model'].apply(lambda x: x['three_backtrack'])
    })
    
    # Save each dataframe to a separate parquet file
    optimal_df.to_parquet('/home/users/hc387/data/advanced_geometry/sft_backtrack/qwen-instruct_optimal.parquet')
    one_backtrack_df.to_parquet('/home/users/hc387/data/advanced_geometry/sft_backtrack/qwen-instruct_one_backtrack.parquet')
    two_backtrack_df.to_parquet('/home/users/hc387/data/advanced_geometry/sft_backtrack/qwen-instruct_two_backtrack.parquet')
    three_backtrack_df.to_parquet('/home/users/hc387/data/advanced_geometry/sft_backtrack/qwen-instruct_three_backtrack.parquet')
    
    print("Successfully split data into separate parquet files:")
    print("- optimal_backtrack.parquet")
    print("- one_backtrack.parquet")
    print("- two_backtrack.parquet")
    print("- three_backtrack.parquet")


split_backtrack_data("/home/users/hc387/data/advanced_geometry/sft_backtrack/qwen-instruct_sft.parquet") 