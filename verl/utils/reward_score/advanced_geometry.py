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

def parse_two_numbers(input_str: str) -> tuple[float, float]:
    if not (input_str.startswith("(") and input_str.endswith(")")):
        return
    
    content = input_str[1:-1]
    parts = content.split(",")
    
    if len(parts) != 2:
        return
    
    num1 = parts[0].strip()
    num2 = parts[1].strip()
    
    return num1, num2


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
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

def compute_score_orthocenter(solution_str, ground_truth, method='strict', format_score=0., score=1.):
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
        return 0
    else:
        answer_num1, answer_num2 = parse_two_numbers(answer)
        # print(f"Answer numbers: {answer_num1}, {answer_num2}")
        ground_truth_num1, ground_truth_num2 = parse_two_numbers(ground_truth)
        # print(f"Ground truth numbers: {ground_truth_num1}, {ground_truth_num2}")
        if answer_num1 is None or answer_num2 is None:
            if do_print:
                print("Answers not found")
            return 0
        elif answer_num1 == ground_truth_num1 and answer_num2 == ground_truth_num2:
            if do_print:
                print(f"Correct answer, score: {score}")
            return score
        else:
            if do_print:
                print(f"Incorrect answer, score: {format_score}")