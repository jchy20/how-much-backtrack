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

import re
import random
import ast
import operator


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:" or "<|im_start|>assistant"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    
    # Check if <think> and </think> appear exactly once and in the correct order
    think_starts = solution_str.count("<think>")
    think_ends = solution_str.count("</think>")
    if think_starts != 1 or think_ends != 1:
        return None
    
    think_start_pos = solution_str.find("<think>")
    think_end_pos = solution_str.find("</think>")
    if think_start_pos > think_end_pos:
        return None
    
    think_content = solution_str[think_start_pos + len("<think>"):think_end_pos]
    if not think_content.strip():
        return None
    # Check if <answer> and </answer> appear exactly once and in the correct order
    answer_starts = solution_str.count("<answer>")
    answer_ends = solution_str.count("</answer>")
    if answer_starts != 1 or answer_ends != 1:
        return None
    
    answer_start_pos = solution_str.find("<answer>")
    answer_end_pos = solution_str.find("</answer>")
    if answer_start_pos > answer_end_pos:
        return None
    
    # Check that answer is not inside think
    if think_end_pos > answer_start_pos:
        return None
    # Extract the answer
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        return final_answer
    else:
        return None

def parse_two_numbers(input_str: str):
    try:
        if not (input_str.startswith("(") and input_str.endswith(")")):
            return None, None
        
        content = input_str[1:-1]
        
        # Handle different separator formats: comma with or without space, or just space
        if "," in content:
            parts = content.split(",")
        else:
            parts = content.split()
        
        if len(parts) != 2:
            return None, None
        
        num1 = parts[0].strip()
        num2 = parts[1].strip()
        
        return num1, num2
    except:
        return None, None

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    answer = extract_solution(solution_str=solution_str)
    # print(f"Answer: {answer}")
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Extracted answer: {answer}")
        print(f"Ground truth: {ground_truth}")
        print("\n")

    if answer is None:
        if do_print:
            print("No answer found")
        return 0
    else:
        if answer == ground_truth:
            if do_print:
                print(f"Correct answer, score: {score}")
            return score
        else:
            if do_print:
                print(f"Incorrect answer, score: {format_score}")
            return format_score

def compute_score_orthocenter(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    answer = extract_solution(solution_str=solution_str)
    gt_num1, gt_num2 = parse_two_numbers(ground_truth)
    # print(f"Answer: {answer}")
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Extracted answer: {answer}")
        print(f"Ground truth: {ground_truth}")
        print("\n")

    if answer is None:
        if do_print:
            print("No answer found, incorrect solution format")
        return 0
    else:
        try:
            answer_num1, answer_num2 = parse_two_numbers(answer)
            if answer_num1 == gt_num1 and answer_num2 == gt_num2:
                if do_print:
                    print(f"Correct answer, score: {score}")
                return score
            else:
                if do_print:
                    print(f"Incorrect answer, score: {format_score}")
                return format_score
        except:
            if do_print:
                print(f"Can't parse the answer to two numbers, format score: {format_score}")
            return format_score
        


########################################################
# For baseline eval only
########################################################
def baseline_extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def baseline_compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    answer = baseline_extract_solution(solution_str=solution_str)
    # print(f"Answer: {answer}")
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Extracted answer: {answer}")
        print(f"Ground truth: {ground_truth}")
        print("\n")

    if answer is None:
        if do_print:
            print("No answer found")
        return 0
    else:
        if answer == ground_truth:
            if do_print:
                print(f"Correct answer, score: {score}")
            return score
        else:
            if do_print:
                print(f"Incorrect answer, score: {format_score}")
            return format_score

def baseline_compute_score_orthocenter(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    answer = baseline_extract_solution(solution_str=solution_str)
    gt_num1, gt_num2 = parse_two_numbers(ground_truth)
    # print(f"Answer: {answer}")
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Extracted answer: {answer}")
        print(f"Ground truth: {ground_truth}")
        print("\n")

    if answer is None:
        if do_print:
            print("No answer found, incorrect solution format")
        return 0
    else:
        try:
            answer_num1, answer_num2 = parse_two_numbers(answer)
            if answer_num1 == gt_num1 and answer_num2 == gt_num2:
                if do_print:
                    print(f"Correct answer, score: {score}")
                return score
            else:
                if do_print:
                    print(f"Incorrect answer, score: {format_score}")
                return format_score
        except:
            if do_print:
                print(f"Can't parse the answer to two numbers, format score: {format_score}")
            return format_score