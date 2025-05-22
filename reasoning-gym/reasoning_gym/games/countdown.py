from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import sympy
from sympy import Symbol, symbols
from sympy.parsing.sympy_parser import parse_expr

from ..factory import ProceduralDataset, register_dataset

QUESTION_FORMAT_TEMPLATE = """{question}
Final answer format instructions:
1. Provide your solution as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.
5. Do not include any other text or formatting.
"""


@dataclass
class CountdownConfig:
    """Configuration for Countdown Number Game task generation"""

    min_numbers: int = 4  # Minimum numbers to provide
    max_numbers: int = 6  # Maximum numbers to provide
    min_value: int = 1  # Minimum value for source numbers
    max_value: int = 100  # Maximum value for source numbers
    min_target: int = 100  # Minimum target value
    max_target: int = 999  # Maximum target value
    operators: tuple = ("+", "-", "*", "/")  # Allowed operators
    shuffle: bool = True  # Whether to shuffle the order of source numbers
    seed: Optional[int] = None
    size: int = 500
    include_error_paths: bool = True  # Whether to include examples with error paths
    error_path_probability: float = 1.0  # Probability of including an error path in an example
    max_backtracks: int = 3  # Maximum number of backtrack behaviors to include

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_numbers > 1, "min_numbers must be greater than 1"
        assert self.max_numbers >= self.min_numbers, "max_numbers must be >= min_numbers"
        assert self.min_value > 0, "min_value must be positive"
        assert self.max_value >= self.min_value, "max_value must be >= min_value"
        assert self.min_target > 0, "min_target must be positive"
        assert self.max_target >= self.min_target, "max_target must be >= min_target"
        assert len(self.operators) > 0, "must specify at least one operator"
        assert all(op in ("+", "-", "*", "/") for op in self.operators), "invalid operator specified"


class CountdownDataset(ProceduralDataset):
    """Generates Countdown Number Game tasks"""

    def __init__(self, config: CountdownConfig):
        self._prompt_templates = [
            "Using the numbers {numbers}, create an expression that equals {target}.\nYou can only use each number once.",
            "Find a way to make {target} using some or all of these numbers: {numbers}.\nEach number can only be used once.",
            "Calculate {target} using the numbers {numbers}.\nEach number may be used at most once.",
        ]
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Countdown Game task

        Returns:
            dict with keys:
                - question: str, the task description with numbers and target
                - answer: str, one possible solution expression
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        # Generate a valid expression and its result
        expression, numbers, target, steps = self._generate_expression(rng)

        # Optionally randomize the order of numbers
        if self.config.shuffle:
            rng.shuffle(numbers)

        numbers_str = ", ".join(map(str, numbers))

        question = rng.choice(self._prompt_templates)
        question = question.format(numbers=numbers_str, target=target)
        
        # Format the combined solution
        formatted_expression = self._format_combined_solution(expression)
        
        # Generate error paths if configured
        error_paths = []
        if self.config.include_error_paths and rng.random() < self.config.error_path_probability:
            error_paths = self._generate_multiple_error_paths(rng, steps, numbers, target)

        return {
            "question": QUESTION_FORMAT_TEMPLATE.format(question=question),
            "answer": expression,
            "metadata": {
                "numbers": numbers,
                "target": target,
                "expression": expression,
                "steps": steps,
                "combined_solution": formatted_expression,
                "error_paths": error_paths
            },
        }

    def _generate_candidate_expression(self, rng: Random, num_terms: int) -> tuple[sympy.Expr, list[int], list[Symbol], list[str]]:
        """Generate a candidate expression with random numbers and operators"""
        # Generate random numbers
        numbers = [rng.randint(self.config.min_value, self.config.max_value) for _ in range(num_terms)]
        
        # Create symbols for building expression
        syms = symbols(f"x:{num_terms}")

        # Build random expression
        expr = syms[0]
        steps = []
        current_value = numbers[0]
        
        for i in range(1, num_terms):
            op = rng.choice(self.config.operators)
            prev_expr = expr  # Store previous expression
            
            if op == "+":
                expr = expr + syms[i]
            elif op == "-":
                expr = expr - syms[i]
            elif op == "*":
                expr = expr * syms[i]
            else:  # division
                # Handle division carefully to ensure integer results
                if numbers[i] != 0:  # Avoid division by zero
                    # Get current value after substituting previous numbers
                    current = int(expr.subs({sym: num for sym, num in zip(syms[:i], numbers[:i])}))
                    # Try each remaining number to find one that divides evenly
                    remaining = [n for n in numbers[i:] if n != 0]
                    rng.shuffle(remaining)  # Randomize order for variety
                    found_divisor = False
                    for div in remaining:
                        if current % div == 0:  # Check if divides evenly
                            numbers[i] = div
                            expr = expr / syms[i]
                            found_divisor = True
                            break
                    if not found_divisor:
                        # If no number divides evenly, fallback to subtraction
                        expr = expr - syms[i]
                        # Update the operator to reflect the actual operation
                        op = "-"
                else:
                    # Fallback to addition for zero
                    expr = expr + syms[i]
                    # Update the operator to reflect the actual operation
                    op = "+"

            current_subs = {sym: num for sym, num in zip(syms[:i+1], numbers[:i+1])}
            new_value = int(expr.subs(current_subs))
            
            # Create step using previous result and current operation
            step = f"{current_value} {op} {numbers[i]} = {new_value}"
            steps.append(step)
            
            # Update current value for next iteration
            current_value = new_value
        
        return expr, numbers, syms, steps

    def _generate_expression(self, rng: Random) -> tuple[str, list[int], int, list[str]]:
        """Generate a valid expression and its result

        Returns:
            Tuple of (expression string, list of numbers used, target value, steps)
        """
        num_terms = rng.randint(self.config.min_numbers, self.config.max_numbers)

        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                expr, numbers, syms, steps = self._generate_candidate_expression(rng, num_terms)
                
                # Substitute actual numbers to get target
                subs = {sym: num for sym, num in zip(syms, numbers)}
                target = int(expr.subs(subs))

                # Convert to string expression
                expr_str = str(expr)
                for i, sym in enumerate(syms):
                    expr_str = expr_str.replace(str(sym), str(numbers[i]))

                # Ensure target is within bounds
                if self.config.min_target <= target <= self.config.max_target:
                    return expr_str, numbers, target, steps
                
            except (ValueError, ZeroDivisionError):
                continue

        raise ValueError(f"Failed to generate valid expression after {max_attempts} attempts")

    def _format_combined_solution(self, expression: str) -> str:
        """Format the combined solution expression for better readability"""
        # Add spaces around operators for better readability
        formatted = expression.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').replace('/', ' / ')
        # Remove any double spaces
        formatted = ' '.join(formatted.split())
        return formatted

# ---------------------------------------------------------------------------
# 1)  _generate_incorrect_steps
# ---------------------------------------------------------------------------
    def _generate_incorrect_steps(
        self,
        rng: Random,
        correct_steps: list[str],
        numbers: list[int],
        target: int,
        start_step: int = 1,
    ) -> tuple[list[str], int]:
        """
        Generate a wrong trace where **every** step i ≥ start_step differs
        from the original step i.

        Constraints
        -----------
        • Each source number appears at most once in the entire wrong trace.
        • Integer division only when it divides evenly; otherwise we fall back to '-'.
        """
        # ---------------- helpers -----------------
        def apply(op: str, a: int, b: int) -> tuple[str, int]:
            """Return (effective_op, result) with integer semantics."""
            if op == "+":  return op, a + b
            if op == "-":  return op, a - b
            if op == "*":  return op, a * b
            if op == "/":
                if b != 0 and a % b == 0:
                    return op, a // b
                return "-", a - b        # safe fallback
            raise ValueError(op)

        def mutate_step(prev_val: int, op_orig: str, num_orig: int) -> tuple[str, int, int]:
            """
            Produce (new_op, new_num, new_val) such that
            (new_op, new_num) ≠ (op_orig, num_orig).
            """
            # --- decide whether to change operator or number -----------------
            mutate_op = rng.random() < 0.5
            new_op, new_num = op_orig, num_orig

            if mutate_op:
                alt_ops = [o for o in self.config.operators if o != op_orig]
                rng.shuffle(alt_ops)
                for cand in alt_ops:
                    eff, val = apply(cand, prev_val, num_orig)
                    if val != apply(op_orig, prev_val, num_orig)[1]:
                        new_op = eff
                        return new_op, new_num, val

            # -------- change operand instead -------------------------------
            fresh_nums = [n for n in numbers if used[n] == 0 and n != num_orig]
            if fresh_nums:
                new_num = rng.choice(fresh_nums)
            else:
                # No fresh number left; we must change operator
                alt_ops = [o for o in self.config.operators if o != op_orig]
                eff, val = apply(alt_ops[0], prev_val, num_orig)
                new_op = eff
                return new_op, new_num, val

            eff, val = apply(op_orig, prev_val, new_num)
            # If that accidentally stayed equal to the original result, flip op
            if val == apply(op_orig, prev_val, num_orig)[1]:
                alt_ops = [o for o in self.config.operators if o != op_orig]
                eff, val = apply(alt_ops[0], prev_val, new_num)
                new_op = eff
            return new_op, new_num, val

        # ------------------------------------------------------------------
        incorrect_steps = correct_steps.copy()

        # Edge case: nothing to mutate
        if len(incorrect_steps) <= start_step:
            return incorrect_steps, target

        # Usage counter for numbers already consumed **before** start_step
        used = {n: 0 for n in numbers}
        for i in range(start_step):
            tok_num = int(correct_steps[i].split()[2])
            used[tok_num] += 1

        # ------------------------------------------------------------------
        prev_val = int(correct_steps[start_step].split()[0])  # value entering start_step
        for i in range(start_step, len(incorrect_steps)):
            op_orig, num_orig = incorrect_steps[i].split()[1:3]
            num_orig = int(num_orig)

            # generate mutated (op, num, val)
            new_op, new_num, new_val = mutate_step(prev_val, op_orig, num_orig)

            # update bookkeeping
            if new_num != num_orig:  # if we swapped operands
                used[num_orig] -= 0           # orig wasn't used yet in wrong trace
                used[new_num]  += 1
            else:
                # operand stayed, but maybe operator changed
                used[num_orig] += 0

            incorrect_steps[i] = f"{prev_val} {new_op} {new_num} = {new_val}"
            prev_val = new_val

        return incorrect_steps, prev_val

# ---------------------------------------------------------------------------
# 2)  _generate_multiple_error_paths
# ---------------------------------------------------------------------------
    def _generate_multiple_error_paths(
        self,
        rng: Random,
        correct_steps: list[str],
        numbers: list[int],
        target: int,
    ) -> list[tuple[list[str], int, int]]:
        """
        Generate 3 **distinct** error traces.
        recovery_step = start_step - 1  (−1 if start_step == 0)
        """
        paths, seen = [], set()
        num_backtracks = 3
        max_global_attempts = 100    # hard cap to avoid infinite loop

        attempts = 0
        while len(paths) < num_backtracks and attempts < max_global_attempts:
            attempts += 1
            start_step = rng.randint(0, len(correct_steps) - 1)

            err_steps, err_val = self._generate_incorrect_steps(rng, correct_steps, numbers, target, start_step)
            fingerprint = tuple(err_steps)

            if fingerprint in seen:
                continue

            seen.add(fingerprint)
            recovery = start_step - 1       # −1 if diverged at 0
            paths.append((err_steps, err_val, recovery))

        if len(paths) < num_backtracks:
            raise RuntimeError(
                f"Could only create {len(paths)} unique error paths "
                "— solution trace is probably too short."
            )
        return paths
    
    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the problem"""
        reward = 0.0
        metadata = entry["metadata"]
        if answer is not None:
            try:
                user_answer = int(parse_expr(answer))
                solved = user_answer == metadata["target"]
                if solved:
                    reward = 1.0
                elif len(answer.strip()) > 0:  # encourage partial solutions
                    reward = 0.05
                else:
                    reward = 0.01
            except:
                reward = 0.01
        return reward


# Register the dataset
register_dataset("countdown", CountdownDataset, CountdownConfig)
