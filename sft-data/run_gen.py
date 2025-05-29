from openai import OpenAI
import pandas as pd
import argparse
from verl.utils.reward_score import multiply, countdown, advanced_geometry, arc_1d, sudoku, color_cube_rotation, zebra_puzzles, list_functions, self_reference
from tqdm import tqdm
import os
import time
import concurrent.futures

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True,
                  help="Path to the model directory")
parser.add_argument("--model_type", type=str, required=True,
                  help="Type of model to use")
parser.add_argument("--eval_dataset_dir", type=str, required=True,
                  help="Directory containing evaluation dataset")
parser.add_argument("--port", type=str, default="8000",
                  help="Port number for the API server")
parser.add_argument("--task_name", type=str, default=None,
                  help="Name of the task being evaluated (for logging)")
parser.add_argument("--output_dir", type=str, default="~/data/sft_data",
                  help="Directory to save output")
parser.add_argument("--save_interval", type=int, default=20000,
                  help="Save data every N samples")
parser.add_argument("--use_together", action="store_true", 
                    help="Use Together API")
parser.add_argument("--batch_size", type=int, default=32,
                  help="Number of examples to process in a single batch")
parser.add_argument("--n", type=int, default=1,
                  help="Number of completions to generate")
parser.add_argument("--job_id", type=int, default=None,
                  help="Job ID")
parser.add_argument("--total_splits", type=int, default=None,
                  help="Total splits")
args = parser.parse_args()

model_path = args.model_path
model_type = args.model_type
eval_dataset_dir = args.eval_dataset_dir
port = args.port
task_name = args.task_name
output_dir = args.output_dir
save_interval = args.save_interval
use_together = args.use_together
batch_size = args.batch_size
n = args.n
job_id = args.job_id
total_splits = args.total_splits

if use_together:
    client = OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
else:
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="EMPTY",
    )

class Generator:
    @staticmethod
    def _select_rm_score_fn(data_source):
        if data_source == 'advanced_geometry/incircle_radius' or data_source == 'advanced_geometry/angle_measure':
            return advanced_geometry.compute_score
        elif data_source == 'advanced_geometry/orthocenter':
            return advanced_geometry.compute_score_orthocenter
        elif data_source == 'arc_1d':
            return arc_1d.compute_score
        elif data_source == 'sudoku':
            return sudoku.compute_score
        elif data_source == 'color_cube_rotation':
            return color_cube_rotation.compute_score
        elif data_source == 'zebra_puzzles':
            return zebra_puzzles.compute_score
        elif data_source == 'list_functions':
            return list_functions.compute_score
        elif data_source == 'self_reference':
            return self_reference.compute_score
        elif "countdown" in data_source:
            return countdown.compute_score
        else:
            raise NotImplementedError

    @staticmethod
    def query_openai(prompts: list[str], model_path: str = model_path, n: int = n):
        try:
            chat_response = client.completions.create(
                model=model_path,
                prompt=prompts,
                max_tokens=8192,
                n = n,
            )
            grouped_completions = [[] for _ in range(len(prompts))]
            for idx, choice in enumerate(chat_response.choices):
                grouped_completions[idx // n].append(choice.text)
            return grouped_completions
        except Exception as e:
            print(f"API request failed: {e}")
            print(f"Waiting for 10 seconds before continuing...")
            time.sleep(10)
            return [[] for _ in range(len(prompts))]
    
    @staticmethod
    def save_data(correct_rows, incorrect_rows, incorrect_format_rows, model_type, task_name, output_dir, job_id, checkpoint=False, processed_count=None):
        """Save the collected data to parquet files."""
        correct_parquet = pd.DataFrame(correct_rows)
        incorrect_parquet = pd.DataFrame(incorrect_rows)
        incorrect_format_parquet = pd.DataFrame(incorrect_format_rows)

        correct_dir = os.path.join(output_dir, "correct")
        incorrect_dir = os.path.join(output_dir, "incorrect")
        incorrect_format_dir = os.path.join(output_dir, "incorrect_format")

        os.makedirs(correct_dir, exist_ok=True)
        os.makedirs(incorrect_dir, exist_ok=True)
        os.makedirs(incorrect_format_dir, exist_ok=True)

        filename = f'{task_name}_{job_id}'
        if checkpoint:
            if processed_count is not None:
                filename = f'{task_name}_{job_id}_checkpoint_{processed_count}'
            else:
                timestamp = int(time.time())
                filename = f'{task_name}_{job_id}_checkpoint_{timestamp}'
        
        if not correct_parquet.empty:
            correct_parquet.to_parquet(os.path.join(correct_dir, f'{filename}.parquet'))
            print(f"Saved {len(correct_parquet)} correct samples to {os.path.join(correct_dir, f'{filename}.parquet')}")
        if not incorrect_parquet.empty:
            incorrect_parquet.to_parquet(os.path.join(incorrect_dir, f'{filename}.parquet'))
            print(f"Saved {len(incorrect_parquet)} incorrect samples to {os.path.join(incorrect_dir, f'{filename}.parquet')}")
        if not incorrect_format_parquet.empty:
            incorrect_format_parquet.to_parquet(os.path.join(incorrect_format_dir, f'{filename}.parquet'))
            print(f"Saved {len(incorrect_format_parquet)} incorrect format samples to {os.path.join(incorrect_format_dir, f'{filename}.parquet')}")
   
    @staticmethod
    def process_batch(rows, model_path: str = model_path, n: int = n):
        """Process a single example with the API and evaluate the results."""
        prompts = []
        ground_truths = []
        data_sources = []
        for i in range(len(rows)):
            prompts.append(rows.iloc[i]['prompt'][0]['content'])
            ground_truths.append(rows.iloc[i]['reward_model']['ground_truth'])
            data_sources.append(rows.iloc[i]['data_source'])
        
        # Get completions from the API
        completions = Generator.query_openai(prompts=prompts, model_path=model_path, n=n)
        
        if not completions:
            print(f"Skipping sample due to API failure")
            return [], [], []
        
        # Process each completion
        correct_rows = []
        incorrect_rows = []
        incorrect_format_rows = []
        
        for i, completions_group in enumerate(completions):
            ground_truth = ground_truths[i]
            prompt = prompts[i]
            compute_score_fn = Generator._select_rm_score_fn(data_sources[i])

            for j, completion in enumerate(completions_group):
                prompt_response = prompt + completion

                # Check format correctness first
                if self_reference.extract_solution(prompt_response) is None:
                    incorrect_format_rows.append({'prompt': prompt, 'completion': completion})
                    continue
            
                # Compute score
                score = compute_score_fn(solution_str=prompt_response, ground_truth=ground_truth)
                
                # Add to appropriate list
                new_pair = {'prompt': prompt, 'completion': completion}
                if score == 1.0:
                    correct_rows.append(new_pair)
                else:
                    incorrect_rows.append(new_pair)
        
        return correct_rows, incorrect_rows, incorrect_format_rows
    
    @staticmethod
    def generate(dataset_dir: str = eval_dataset_dir, batch_size: int = batch_size, model_type: str = model_type, model_path: str = model_path, n: int = n, task_name: str = task_name, output_dir: str = output_dir, save_interval: int = save_interval, total_splits: int = total_splits):
        df = pd.read_parquet(dataset_dir)
        num_sample = len(df) // total_splits  # Use integer division
        start_idx = job_id * num_sample
        end_idx = min((job_id + 1) * num_sample, len(df))  # Ensure we don't exceed DataFrame length
        df = df.iloc[start_idx:end_idx]
        
        correct_rows = []
        incorrect_rows = []
        incorrect_format = []

        correct_count = 0
        incorrect_count = 0
        incorrect_format_count = 0

        processed_count = 0
        
        # Create progress bar
        pbar = tqdm(range(0, len(df), batch_size), desc=f"Generating for {task_name} [Correct: 0 | Incorrect: 0]")

        # Process batches with tqdm for progress tracking
        for i in pbar:
            try:
                batch = df.iloc[i:i+batch_size]
                batch_correct_rows, batch_incorrect_rows, batch_incorrect_format = Generator.process_batch(batch, model_path=model_path, n=n)
                correct_rows.extend(batch_correct_rows)
                incorrect_rows.extend(batch_incorrect_rows)
                incorrect_format.extend(batch_incorrect_format)
                correct_count += len(batch_correct_rows)
                incorrect_count += len(batch_incorrect_rows)
                incorrect_format_count += len(batch_incorrect_format)
                processed_count += len(batch)
                
                # Update progress bar description with current stats
                pbar.set_description(f"Generating for {task_name} [Correct: {correct_count} | Incorrect: {incorrect_count}]")
                
                if processed_count % save_interval == 0:
                    print(f"\nSaving checkpoint after processing {processed_count}/{len(df)} samples...")
                    Generator.save_data(correct_rows, incorrect_rows, incorrect_format, model_type, task_name, output_dir, job_id, checkpoint=True, processed_count=processed_count)
                    print("Checkpoint saved.")
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        # Final save at the end
        Generator.save_data(correct_rows, incorrect_rows, incorrect_format, model_type, task_name, output_dir, job_id)

        # return counts
        return correct_count, incorrect_count, incorrect_format_count
    

if __name__ == "__main__":
    correct_count, incorrect_count, incorrect_format_count = Generator.generate()
    print(f"Correct trajectories for {task_name} is: {correct_count}. Incorrect trajectories is: {incorrect_count}. Incorrect format is: {incorrect_format_count}")