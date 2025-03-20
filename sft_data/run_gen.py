from openai import OpenAI
import pandas as pd
import argparse
from verl.utils.reward_score import multiply, countdown, advanced_geometry, arc_1d, sudoku, color_cube_rotation, zebra_puzzles, list_functions, self_reference
from tqdm import tqdm
import os

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
parser.add_argument("--output_dir", type=str, default="/home/users/hc387/data/sft_data",
                  help="Directory to save output")
args = parser.parse_args()

model_path = args.model_path
model_type = args.model_type
eval_dataset_dir = args.eval_dataset_dir
port = args.port
task_name = args.task_name
output_dir = args.output_dir

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
    def query_openai(prompt: str, model_path: str = model_path):
        chat_response = client.completions.create(
            model=model_path,
            prompt=prompt,
            max_tokens=4096,
            n = 5,
        )
        completions = [choice.text for choice in chat_response.choices]
        # prompt_responses = [prompt + response for response in only_responses]
        return completions
        
    @staticmethod
    def generate(dataset_dir: str = eval_dataset_dir):
        df = pd.read_parquet(dataset_dir)
        correct_rows = []
        incorrect_rows = []

        correct_count = 0
        incorrect_count = 0

        pbar = tqdm(df.iterrows(), total=len(df), desc="Evaluating", unit="sample")
        for index, row in pbar:
            prompt = row['prompt'][0]['content']
            ground_truth = row['reward_model']['ground_truth']
            completions = Generator.query_openai(prompt)
            compute_score_fn = Generator._select_rm_score_fn(row['data_source'])

            prompt_responses = [prompt + completion for completion in completions]
            
            for idx, prompt_response in enumerate(prompt_responses):
                if self_reference.extract_solution(prompt_response) is None:
                    continue
                

                score = compute_score_fn(solution_str=prompt_response, ground_truth=ground_truth)
                new_pair = {'prompt': prompt, 'completion': completions[idx]}
                if score == 1.0:
                    correct_count += 1
                    correct_rows.append(new_pair)
                else:
                    incorrect_count += 1
                    incorrect_rows.append(new_pair)

            pbar.set_postfix(correct_count=correct_count, incorrect_count=incorrect_count)

        correct_parquet = pd.DataFrame(correct_rows)
        incorrect_parquet = pd.DataFrame(incorrect_rows)

        correct_dir = os.path.join(output_dir, model_type, "correct")
        incorrect_dir = os.path.join(output_dir, model_type, "incorrect")

        os.makedirs(correct_dir, exist_ok=True)
        os.makedirs(incorrect_dir, exist_ok=True)

        if not correct_parquet.empty:
            correct_parquet.to_parquet(os.path.join(correct_dir, f'{task_name}.parquet'))
        if not incorrect_parquet.empty:
            incorrect_parquet.to_parquet(os.path.join(incorrect_dir, f'{task_name}.parquet'))


        # return counts
        return correct_count, incorrect_count
    

if __name__ == "__main__":
    correct_count, incorrect_count = Generator.generate()
    task_identifier = task_name if task_name else eval_dataset_dir
    print(f"Correct trajectories for {task_identifier} is: {correct_count}. Incorrect trajectories is: {incorrect_count}")
