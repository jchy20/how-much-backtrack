from openai import OpenAI
import pandas as pd
import argparse
from verl.utils.reward_score import multiply, countdown, advanced_geometry, arc_1d, sudoku, color_cube_rotation, zebra_puzzles, list_functions, self_reference
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True,
                  help="Path to the model directory")
parser.add_argument("--eval_dataset_dir", type=str, required=True,
                  help="Directory containing evaluation dataset")
parser.add_argument("--port", type=str, default="8000",
                  help="Port number for the API server")
parser.add_argument("--task_name", type=str, default=None,
                  help="Name of the task being evaluated (for logging)")
args = parser.parse_args()

model_path = args.model_path
eval_dataset_dir = args.eval_dataset_dir
port = args.port
task_name = args.task_name



class Evaluator:
    @staticmethod
    def query_openai(prompt: str, model_path: str = model_path, port: str = port):
        client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
        )

        chat_response = client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "user", "content": prompt},
            ],
            # n = 5,
        )
        # for choice in chat_response.choices:
        #     if self_reference.baseline_compute_score(choice.message.content) is not None:
        #         return choice.message.content
        return chat_response.choices[0].message.content

    @staticmethod
    def _select_rm_score_fn(data_source):
        if data_source == 'advanced_geometry/incircle_radius' or data_source == 'advanced_geometry/angle_measure':
            return advanced_geometry.baseline_compute_score
        elif data_source == 'advanced_geometry/orthocenter':
            return advanced_geometry.baseline_compute_score_orthocenter
        elif data_source == 'arc_1d':
            return arc_1d.baseline_compute_score
        elif data_source == 'sudoku':
            return sudoku.baseline_compute_score
        elif data_source == 'color_cube_rotation':
            return color_cube_rotation.baseline_compute_score
        elif data_source == 'zebra_puzzles':
            return zebra_puzzles.baseline_compute_score
        elif data_source == 'list_functions':
            return list_functions.baseline_compute_score
        elif data_source == 'self_reference':
            return self_reference.baseline_compute_score
        elif "multiply" in data_source or "arithmetic" in data_source:
            return multiply.baseline_compute_score
        elif "countdown" in data_source:
            return countdown.baseline_compute_score
        else:
            raise NotImplementedError
        

    @staticmethod
    def evaluate(dataset_dir: str = eval_dataset_dir):
        df = pd.read_parquet(dataset_dir)
        format_correct_count = 0
        correct_count = 0
        total_count = 0
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating", unit="sample"):
            prompt = row['prompt'][0]['content']
            ground_truth = row['reward_model']['ground_truth']
            response = Evaluator.query_openai(prompt)
            response = "<|im_start|>assistant: " + response
            compute_score_fn = Evaluator._select_rm_score_fn(row['data_source'])
            if self_reference.baseline_extract_solution(response) is not None:
                format_correct_count += 1
            score = compute_score_fn(solution_str=response, ground_truth=ground_truth)

            if score == 1.0:
                correct_count += 1
            total_count += 1

        # return accuracy
        return correct_count / total_count, format_correct_count / total_count
    

if __name__ == "__main__":
    accuracy, format_accuracy = Evaluator.evaluate()
    task_identifier = task_name if task_name else eval_dataset_dir
    print(f"Task accuracy for {task_identifier} is: {accuracy}\nFormat accuracy for {task_identifier} is: {format_accuracy}")

