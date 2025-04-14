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
parser.add_argument("--baseline", type=lambda x: x.lower() == 'true', default=False,
                  help="Whether to use baseline evaluation")
parser.add_argument("--batch_size", type=int, default=16,
                  help="Batch size for evaluation")
args = parser.parse_args()

model_path = args.model_path
eval_dataset_dir = args.eval_dataset_dir
port = args.port
task_name = args.task_name
baseline = args.baseline
batch_size = args.batch_size

class Evaluator:
    @staticmethod
    def query_openai(prompts: list[str], model_path: str = model_path, port: str = port):
        client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
        )

        # chat_response = client.chat.completions.create(
        #     model=model_path,
        #     messages=[
        #         {"role": "user", "content": prompt},
        #     ],
        #     # n = 5,
        # )

        chat_response = client.completions.create(
            model=model_path,
            prompt=prompts,
            max_tokens=4096,
            # n = 5,
        )
        completions = []
        for i, prompt in enumerate(prompts):
            completions.append(prompt + chat_response.choices[i].text)

        return completions

        # for choice in chat_response.choices:
        #     if self_reference.baseline_compute_score(choice.message.content) is not None:
        #         return choice.message.content
        
        # response = prompt + chat_response.choices[0].text
        # return response

    @staticmethod
    def _select_rm_score_fn(data_source, baseline=False):
        if baseline:
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
            elif "countdown" in data_source:
                return countdown.baseline_compute_score
            else:
                raise NotImplementedError
        else:
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
    def process_batch(rows, model_path: str = model_path):
        """Process a single example with the API and evaluate the results."""
        prompts = []
        ground_truths = []
        data_sources = []
        for i in range(len(rows)):
            prompts.append(rows.iloc[i]['prompt'][0]['content'])
            ground_truths.append(rows.iloc[i]['reward_model']['ground_truth'])
            data_sources.append(rows.iloc[i]['data_source'])
        
        completions = Evaluator.query_openai(prompts=prompts, model_path=model_path, port=port)
        
        correct_count = 0
        format_correct_count = 0
        total_count = len(completions)
        
        for i, completion in enumerate(completions):
            ground_truth = ground_truths[i]
            compute_score_fn = Evaluator._select_rm_score_fn(data_sources[i], baseline=baseline)
            if self_reference.extract_solution(completion) is not None:
                format_correct_count += 1
            if compute_score_fn(completion, ground_truth) == 1.0:
                correct_count += 1
        
        return correct_count, format_correct_count, total_count

    @staticmethod
    def evaluate(dataset_dir: str = eval_dataset_dir, baseline: bool = baseline, batch_size: int = batch_size):
        df = pd.read_parquet(dataset_dir)
        correct_count = 0
        format_correct_count = 0
        total_count = 0

        pbar = tqdm(range(0, len(df), batch_size), desc=f"Evaluating for {task_name} [Correct: 0 | Format Correct: 0]")

        for i in pbar:
            batch = df.iloc[i:i+batch_size]
            batch_correct, batch_format_correct, batch_total = Evaluator.process_batch(batch, model_path=model_path)
            correct_count += batch_correct
            format_correct_count += batch_format_correct
            total_count += batch_total

            pbar.set_description(f"Evaluating for {task_name} [Correct: {correct_count/total_count} | Format Correct: {format_correct_count/total_count}]")
            
        return correct_count / total_count, format_correct_count / total_count
    

if __name__ == "__main__":
    accuracy, format_accuracy = Evaluator.evaluate()
    task_identifier = task_name if task_name else eval_dataset_dir
    print(f"Task accuracy for {task_identifier} is: {accuracy}. Format accuracy is: {format_accuracy}")

