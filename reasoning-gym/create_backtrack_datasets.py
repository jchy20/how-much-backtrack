import pandas as pd
import random
import os
from datasets import Dataset

# Read the original dataset
input_path = '/home/users/hc387/data/countdown/sft_backtrack/qwen-instruct_sft_optim.parquet'
output_dir = '/home/users/hc387/data/countdown/sft_backtrack'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the dataset
df = pd.read_parquet(input_path)

def create_backtrack_trajectory(example, num_backtracks):
    """
    Create a trajectory with the specified number of backtracks.
    
    Args:
        example: The example to process
        num_backtracks: Number of backtracks to inject (1, 2, or 3)
        
    Returns:
        A string containing the trajectory with backtracks
    """
    # Extract the correct steps and error paths
    correct_steps = example['reward_model']['correct_step']
    correct_steps = correct_steps.tolist()
    
    # Extract error paths
    error_paths = []
    for i in range(1, 4):
        steps = example['reward_model'][f'error_path_{i}_steps']
        value = example['reward_model'][f'error_path_{i}_value']
        recovery = example['reward_model'][f'error_path_{i}_recovery']
        if steps is not None and len(steps) > 0:
            error_paths.append((steps, value, recovery))
    
    # If we don't have enough error paths, return the original trajectory
    if len(error_paths) < num_backtracks:
        return "Let's think step by step. " + example['reward_model']['reasoning_trajectory'] + "This matches the problem statement. This is the solution.\n</think>\n\n<answer>" + example['reward_model']['reasoning_final_answer'] + "</answer>"
    
    # Randomly select error paths
    selected_paths = random.sample(error_paths, num_backtracks)
    
    # Sort by recovery step to ensure correct order
    selected_paths.sort(key=lambda x: x[2])
    
    curr_incorrect_steps = 0
    curr_correct_steps = 0


    trajectory = ""
    for i, (error_steps, error_value, recovery_step) in enumerate(selected_paths):

        error_steps = error_steps.tolist()

        # Add correct steps up to first error step
        for correct_step_idx in range(curr_correct_steps, recovery_step+1):
            curr_correct_steps += 1
            trajectory += f"Step {curr_correct_steps}: {correct_steps[correct_step_idx]}. "

        curr_incorrect_steps = curr_correct_steps
        
        # Add error steps
        for error_step_idx in range(recovery_step+1, len(error_steps)):
            curr_incorrect_steps += 1
            trajectory += f"Step {curr_incorrect_steps}: {error_steps[error_step_idx]}. "

        if curr_correct_steps == 0:
            trajectory += f"Wait, this doesn't lead to the correct solution. Let me restart.\n"
        else:
            trajectory += f"Wait, this doesn't lead to the correct solution. {error_value} is not the correct answer. Let me go back to step {curr_correct_steps} and keep thinking from there.\n"
    
    for i in range(curr_correct_steps, len(correct_steps)):
        trajectory += f"Step {i+1}: {correct_steps[i]}. "
        
    trajectory += "This matches the problem statement. This is the solution.\n</think>\n\n<answer>" + example['reward_model']['reasoning_final_answer'] + "</answer>"
    return trajectory
    

# Create datasets with different numbers of backtracks
for num_backtracks in [1, 2, 3]:
    processed_data = []
    
    for i in range(len(df)):
        example = df.iloc[i]
        prompt = example['prompt'][0]['content']
        completion = create_backtrack_trajectory(example, num_backtracks)
        
        processed_data.append({
            'prompt': prompt,
            'completion': completion
        })
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    # Save to parquet
    output_path = os.path.join(output_dir, f'qwen-instruct_sft_{num_backtracks}_backtrack.parquet')
    processed_df.to_parquet(output_path)
    
    print(f"Processed {len(processed_data)} entries with {num_backtracks} backtracks and saved to {output_path}") 