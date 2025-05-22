"""Sudoku puzzle generator"""

import copy
from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Any, Optional
from typing import List
import random
from collections import defaultdict


from ..factory import ProceduralDataset, register_dataset

class Node:
    __slots__ = ("row", "col", "candidates", "reasoning", "tried", "children", "parent", "board")

    def __init__(self, row: int, col: int, candidates: list[int], reasoning: str, parent: "Node | None" = None, board: list[list[int]] = None):
        self.row = row
        self.col = col
        self.candidates = candidates
        self.reasoning = reasoning
        self.tried: list[int] = []
        self.children: dict[int, "Node"] = {}
        self.parent = parent
        self.board = board

@dataclass
class SudokuConfig:
    """
    Configuration for sudoku puzzle generation
    Puzzle generation can be a bit slower for puzzles with a high (~60+) number of empty cells
    """

    min_empty: int = 30  # Minimum number of empty cells
    max_empty: int = 50  # Maximum number of empty cells
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        # 81 - 64 = 17, the minimum number of clues required for 9x9 Sudoku to have a unique solution
        assert 0 <= self.min_empty <= 64, "min_empty must be between 0 and 64"
        assert self.min_empty <= self.max_empty <= 64, "max_empty must be between min_empty and 64"


class SudokuDataset(ProceduralDataset):
    """Generates sudoku puzzles with configurable difficulty"""

    def __init__(self, config: SudokuConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __len__(self) -> int:
        return self.config.size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self.config.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def _is_valid(self, board: list[list[int]], row: int, col: int, num: int) -> bool:
        """Check if number can be placed at position"""
        # Check row
        if num in board[row]:
            return False

        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        return True

    def _get_possible_values(self, board: list[list[int]], row: int, col: int) -> set[int]:
        """Get all possible values for a cell."""
        row_values = set(board[row])
        row_missing = set(range(1, 10)) - row_values
        col_values = set(board[i][col] for i in range(9))
        col_missing = set(range(1, 10)) - col_values

        # Get filled values in the current 3x3 subgrid
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box_values = set()
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                box_values.add(board[i][j])
        box_missing = set(range(1, 10)) - box_values
        candidates = sorted(row_missing & col_missing & box_missing)

        # used_values = row_values | col_values | box_values
        reasoning = f"Based on the current board, row {row+1} is missing {sorted(row_missing)}, column {col+1} is missing {sorted(col_missing)}, and the 3x3 box is {sorted(box_missing)}. Taking the intersection of these sets, the possible values for cell ({row+1},{col+1}) are {candidates}."
        return candidates, reasoning

    def _solve(self, board: list[list[int]]) -> bool:
        """Solve sudoku using backtracking"""
        empty = self._find_empty(board)
        if not empty:
            return True

        row, col = empty
        candidates, reasoning = self._get_possible_values(board, row, col)
        for num in candidates:
            board[row][col] = num
            if self._solve(board):
                return True
            board[row][col] = 0
        return False

    def build_solution_tree(self, puzzle: list[list[int]]) -> Node:
        from copy import deepcopy
        board = deepcopy(puzzle)

        # (optional) verify solvable
        sol = deepcopy(puzzle)
        if not self._solve(sol):
            raise ValueError("Puzzle has no solution")

        solved_head = [None]

        def dfs(parent: Node | None) -> Node | None:
            empty = self._find_empty(board)
            if not empty:
                return None

            r, c = empty
            cands, reason = self._get_possible_values(board, r, c)
            node = Node(r, c, cands, reason, parent)
            node.board = [row[:] for row in board]

            # ← **THIS** must be *inside* the for‐loop: 
            for digit in cands:
                node.tried.append(digit)

                # **link** the parent to *this* node via the digit edge
                if parent is not None:
                    parent.children[digit] = node

                # descend
                board[r][c] = digit
                node.board = deepcopy(board)
                child = dfs(node)
                if child is None:
                    # solution found below this edge
                    if solved_head[0] is None:
                        solved_head[0] = node
                    return None

                # attach the subtree of the failed branch
                node.children[digit] = child
                board[r][c] = 0

            return node

        root = Node(-1, -1, [], "root")
        dfs(root)
        return root, solved_head[0]

    def _walk_to_contradiction(self, start: Node, lines: list[str]) -> None:
        """
        Depth-first walk that always picks the *first* tried digit at each node
        (deterministic) until we reach a node whose tried == [].
        Appends verbalised reasoning to *lines*.
        """
        rng = Random()
        cur = start
        while True:
            if not cur.tried:                                  # no candidates ⇒ dead end
                lines.append(
                    f"Looking at the cell at ({cur.row+1},{cur.col+1}). "
                    f"{cur.reasoning} "
                    f"Wait, we are stuck at cell ({cur.row+1},{cur.col+1}). There are no possible values for this cell. I need to go back to one of previous cells."
                )
                return
            d = rng.choice(cur.tried)                                  # last digit tested here
            lines.append(
                f"Looking at the cell at ({cur.row+1},{cur.col+1}). "
                f"{cur.reasoning} "
                f"Let's try to fill cell ({cur.row+1},{cur.col+1}) with {d}."
            )
            nxt = cur.children.get(d)
            if nxt is None:                                    # leaf without children
                lines.append("No more children for this node. I need to go back to one of previous cells.")
                return
            cur = nxt
    
    def explain_with_detours_from_tree(self, puzzle: list[list[int]], k: int = 1, solved_node: Node | None = None, solution_str: str | None = None, seed: int | None = None) -> str:

        cur = solved_node
        # 3-A. collect optimal path nodes (root → solved_head)
        path: list[Node] = []
        while cur.parent is not None:
            path.append(cur)      # exclude synthetic root
            cur = cur.parent
        path.reverse()                   # top-down order
       
        # 3-B. pick ≤ k detour points among nodes with ≥ 2 tried digits
        rng = Random(seed)
        candidates = [
            (node, d)
            for node in path
            for d in node.tried[:-1]            # exclude the last (correct) digit
        ]

        k = min(k, len(candidates))
        detour_pairs: list[tuple[Node, int]] = rng.sample(candidates, k)
        detours_for_node: dict[Node, list[int]] = defaultdict(list)
        for node, wrong_digit in detour_pairs:
            detours_for_node[node].append(wrong_digit)


        # 3-C. walk the path, injecting detours when scheduled
        lines: list[str] = []
        print_board_counter = 0
        for node in path:
            correct = node.tried[-1]

            if node in detours_for_node:
                for wrong_digit in detours_for_node[node]:
                    wrong_branch = node.children[wrong_digit]

                    lines.append(
                        f"Looking at the cell at ({node.row+1},{node.col+1}). "
                        f"{node.reasoning} "
                        f"Let's try to fill cell ({node.row+1},{node.col+1}) with {wrong_digit}."
                    )
                    self._walk_to_contradiction(wrong_branch, lines)
                    temp_board = [row[:] for row in node.board]
                    temp_board[node.row][node.col] = 0
                    lines.append(
                        f"Back at ({node.row+1},{node.col+1}); erase {wrong_digit}. The current board is:\n{self._board_to_string(temp_board)}"
                    )
                wrong_digits = detours_for_node[node]
                lines.append(
                    f"Out of all the possible values {node.candidates} for cell ({node.row+1},{node.col+1}), I tried {wrong_digits}. Let me try to fill the cell with {correct}."
                )
            
            else:
                print_board_counter += 1
                if print_board_counter % 10 == 0:
                    lines.append(
                        f"Looking at the cell at ({node.row+1},{node.col+1}). "
                        f"{node.reasoning} "
                        f"Let's try to fill cell ({node.row+1},{node.col+1}) with {correct}. The current board is:\n{self._board_to_string(node.board)}"
                    )
                else:
                    lines.append(
                        f"Looking at the cell at ({node.row+1},{node.col+1}). "
                        f"{node.reasoning} "
                        f"Let's try to fill cell ({node.row+1},{node.col+1}) with {correct}."
                    )
        
        lines = "\n".join(lines)
        lines += "\n</think>\n\n"
        lines += "<answer>\n"
        lines += solution_str
        lines += "\n</answer>"
        return lines

    def _find_empty(self, board: list[list[int]]) -> Optional[tuple[int, int]]:
        """Find an empty cell"""
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _generate_solved_board(self, rng: Random) -> list[list[int]]:
        """Generate a complete solved sudoku board"""
        board = [[0] * 9 for _ in range(9)]

        # Fill diagonal boxes first (they are independent)
        for i in range(0, 9, 3):
            nums = list(range(1, 10))
            rng.shuffle(nums)
            pos = 0
            for r in range(i, i + 3):
                for c in range(i, i + 3):
                    board[r][c] = nums[pos]
                    pos += 1

        # Solve the rest
        self._solve(board)
        return board

    def _count_solutions(self, board: list[list[int]], limit: int = 2) -> int:
        """Count the number of solutions for a given board"""

        def _get_min_possibilities_cell(board: list[list[int]]) -> Optional[tuple[int, int, set[int]]]:
            """
            Get the cell with the lowest number of possibilities.
            Returns None if the board is already solved.
            """
            min_possibilities = 10
            min_cell = None
            min_values = None

            for i in range(9):
                for j in range(9):
                    if board[i][j] == 0:
                        possible, _ = self._get_possible_values(board, i, j)
                        if len(possible) < min_possibilities:
                            min_possibilities = len(possible)
                            min_cell = (i, j)
                            min_values = possible
                            if min_possibilities == 1:
                                return (*min_cell, min_values)

            return (*min_cell, min_values) if min_cell else None

        def _count_solutions_helper(board: list[list[int]]) -> int:
            cell_info = _get_min_possibilities_cell(board)
            if not cell_info:
                return 1

            row, col, possible_values = cell_info
            count = 0
            for num in possible_values:
                board[row][col] = num
                count += _count_solutions_helper(board)
                if count >= limit:
                    return count
                board[row][col] = 0
            return count

        return _count_solutions_helper(board)

    def _create_puzzle(self, solved_board: list[list[int]], num_empty: int, rng: Random) -> list[list[int]]:
        """Create puzzle by removing numbers from solved board"""
        puzzle = [row[:] for row in solved_board]
        cells = [(i, j) for i in range(9) for j in range(9)]
        rng.shuffle(cells)
        num_removed = 0

        for i, j in cells:
            saved = puzzle[i][j]
            puzzle[i][j] = 0
            puzzle_copy = copy.deepcopy(puzzle)
            # Check if removing this clue breaks uniqueness
            if self._count_solutions(puzzle_copy) > 1:
                puzzle[i][j] = saved
            else:
                num_removed += 1
                if num_removed == num_empty:
                    break

        return puzzle

    def _board_to_string(self, board: list[list[int]]) -> str:
        """Convert board to string representation"""
        return "\n".join(" ".join(str(x) if x != 0 else "_" for x in row) for row in board)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single sudoku puzzle"""
        rng = Random(self.seed + idx)

        # Generate solved board
        solved_board = self._generate_solved_board(rng)

        # Create puzzle by removing numbers
        num_empty = rng.randint(self.config.min_empty, self.config.max_empty)
        puzzle = self._create_puzzle(solved_board, num_empty, rng)
        tree_root, node = self.build_solution_tree(puzzle)
        # reasoning = []
        # cur = node
        # print_board_counter = 0
        # while cur.parent is not None:                 # stop at the synthetic root
        #     print_board_counter += 1
        #     if print_board_counter % 10 == 0:
        #         reasoning.append(
        #             f"Looking at the cell at ({cur.row+1},{cur.col+1}). "
        #             f"{cur.reasoning} "
        #             f"Let's try to fill cell ({cur.row+1},{cur.col+1}) with {cur.tried[-1]}. "
        #             f"The current board is:\n{self._board_to_string(cur.board)}"
        #         )
        #     else:
        #         reasoning.append(
        #             f"Looking at the cell at ({cur.row+1},{cur.col+1}). "
        #             f"{cur.reasoning} "
        #             f"Let's try to fill cell ({cur.row+1},{cur.col+1}) with {cur.tried[-1]}."
        #         )
        #     cur = cur.parent


        # reasoning.reverse()
        # reasoning = "\n".join(reasoning)
        # reasoning += "\n</think>\n\n"
        # reasoning += "<answer>\n"
        # reasoning += self._board_to_string(solved_board)
        # reasoning += "\n</answer>"

        # Format as strings
        puzzle_str = self._board_to_string(puzzle)
        solution_str = self._board_to_string(solved_board)
        optimal = self.explain_with_detours_from_tree(puzzle=puzzle, k=0, solved_node=node, solution_str=solution_str, seed=rng.randint(0, 1000000))
        one_backtrack = self.explain_with_detours_from_tree(puzzle=puzzle, k=1, solved_node=node, solution_str=solution_str, seed=rng.randint(0, 1000000))
        three_backtrack = self.explain_with_detours_from_tree(puzzle=puzzle, k=3, solved_node=node, solution_str=solution_str, seed=rng.randint(0, 1000000))
        five_backtrack = self.explain_with_detours_from_tree(puzzle=puzzle, k=5, solved_node=node, solution_str=solution_str, seed=rng.randint(0, 1000000))
        ten_backtrack = self.explain_with_detours_from_tree(puzzle=puzzle, k=10, solved_node=node, solution_str=solution_str, seed=rng.randint(0, 1000000))

        

        question = (
            f"Solve this Sudoku puzzle:\n{puzzle_str}\n"
            "Respond with only your answer, formatted as the puzzle, a 9x9 grid with numbers separated by spaces, and rows separated by newlines."
        )

        return {
            "question": question,
            "answer": solution_str,
            "metadata": {"puzzle": puzzle, "solution": solved_board, "num_empty": num_empty, "optimal": optimal, "one_backtrack": one_backtrack, "three_backtrack": three_backtrack, "five_backtrack": five_backtrack, "ten_backtrack": ten_backtrack},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if not answer:
            return 0.0

        oracle_answer = entry["answer"]
        metadata = entry["metadata"]
        solution: list[list[int]] = metadata["solution"]
        board_size: int = len(solution[0])

        # 1. match answer without trailing whitespaces
        answer_stripped = "\n".join(l.rstrip() for l in answer.split("\n"))
        oracle_answer_stripped = "\n".join(l.rstrip() for l in oracle_answer.split("\n"))

        if answer_stripped == oracle_answer_stripped:
            reward = 1.0
        else:
            # 2. accept answers with correct numeric sequence (ignoring non-numeric characters)
            row = 0
            num_matching = 0
            for ln in answer.split("\n"):
                if row >= len(solution):
                    break
                numbers = [int(c) for c in ln if c in "123456789"]
                if len(numbers) != board_size:
                    continue  # ignore lines without numbers
                for a, b in zip(solution[row], numbers):
                    if a == b:
                        num_matching += 1
                row += 1

            reward = num_matching / (board_size * board_size)
            reward *= 0.9  # penalty for not using standard format

        if len(answer) > len(oracle_answer):
            reward *= len(oracle_answer) / len(answer)  # penalty for additional length
        return reward


register_dataset("sudoku", SudokuDataset, SudokuConfig)
