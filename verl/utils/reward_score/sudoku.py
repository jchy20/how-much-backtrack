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
import numpy as np

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
    match = re.search(answer_pattern, solution_str)
    if match:
        final_answer = match.group(1).strip()
        return final_answer
    else:
        return None

def extract_numbers(s):
    lines = s.strip().splitlines()
    result = []
    for line in lines:
        result.append([int(x) for x in line.split()])
    return np.array(result)

def check_solution(answer, ground_truth):
    if len(answer) != len(ground_truth):
        return False
    for i in range(len(answer)):
        if len(answer[i]) != len(ground_truth[i]):
            return False
        for j in range(len(answer[i])):
            if answer[i][j] != ground_truth[i][j]:
                return False
    return True

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    ground_truth = ground_truth['solution']
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Solution string: {solution_str}")
        print(f"Extracted answer: {answer}")
        print(f"Ground truth: {ground_truth}")
        print("\n")

    if answer is None:
        return 0
    
    else:
        try:
            answer = extract_numbers(answer)
            if check_solution(answer, ground_truth):
                if do_print:
                    print(f"Correct answer, score: {score}")
                return score
            else:
                if do_print:
                    print(f"Incorrect answer, score: {format_score}")
                return format_score
            
        except:
            if do_print:
                print(f"Can't extract numbers, format score: {format_score}")
            return format_score