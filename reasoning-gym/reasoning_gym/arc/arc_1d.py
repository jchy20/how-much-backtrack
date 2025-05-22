from dataclasses import dataclass
from random import Random
from typing import Optional

from ..dataset import ProceduralDataset
from ..factory import register_dataset


@dataclass
class Arc1DConfig:
    """Configuration for ARC 1D task generation"""

    min_size: int = 10  # Minimum grid size
    max_size: int = 30  # Maximum grid size
    num_train: int = 3  # Number of training examples
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_size >= 8, "min_size must be >= 8"
        assert self.max_size >= self.min_size, "max_size must be >= min_size"
        assert self.num_train > 0, "num_train must be positive"
        assert self.size > 0, "size must be positive"


class Arc1DDataset(ProceduralDataset):
    """
    Generates ARC 1D tasks by randomly selecting from available task generators

    This dataset is a procedural variant of the 1D-ARC dataset which is described in the paper:
    `LLMs and the Abstraction and Reasoning Corpus:  Successes, Failures, and the Importance
    of Object-based Representations` (https://arxiv.org/abs/2305.18354)

    Ilya Sheprut (optozorax) created rust generators for most of the ARC 1d tasks. For
    reasoning-gym rust tasks were machine-converted to python via Sonnet.

    Ilya's original rust code can be found here: https://github.com/optozorax/arc_1d/
    """

    def __init__(self, config: Arc1DConfig):
        from .arc_1d_tasks import ARC_1D_TASKS

        super().__init__(config=config, seed=config.seed, size=config.size)
        self.ARC_1D_TASKS = ARC_1D_TASKS
        self.task_names = list(ARC_1D_TASKS.keys())

    def __getitem__(self, idx: int) -> dict:
        """Generate a single ARC 1D task with training examples

        Args:
            idx: Index of the item to generate

        Returns:
            dict with keys:
                - question: str, the task description and examples
                - answer: str, the expected output format
                - metadata: dict with generation parameters
        """
        # Create deterministic RNG from base seed and idx
        rng = Random(self.seed + idx)

        # Select random task
        gold_task_name = rng.choice(self.task_names)
        gold_task_func, gold_task_kwargs, gold_transform_func, gold_transform_kwargs, gold_reasoning = self.ARC_1D_TASKS[gold_task_name]

        # Generate training examples
        train_examples = []
        size = rng.randint(self.config.min_size, self.config.max_size)

        for _ in range(self.config.num_train):
            example = None
            while example is None:
                example = gold_task_func(rng, size, **gold_task_kwargs)

            train_examples.append(example)

        # Generate test example
        test_example = None
        while test_example is None:
            test_example = gold_task_func(rng, size, **gold_task_kwargs)

        ### template
        template1 = "{instruction} Applying this rule to the test input, we get the output grid:\n"
        template2 = "The outputs match the given example output. Let's apply this rule to the test input and obtain the answer.\n</think>\n\n<answer>{answer}</answer>"
        template3 = "The outputs do not match the given example outputs. Let's try to find another pattern.\n\n"
        
        ### optimal
        optimal = template1.format(instruction=gold_reasoning)
        for i, example in enumerate(train_examples, 1):
            optimal += (f"Example {i}: " + " ".join(str(x) for x in example["output"]) + "\n")
        optimal += template2.format(answer=" ".join(str(x) for x in test_example["output"]))

        ### detours
        temp_task_names = self.task_names.copy()
        temp_task_names.remove(gold_task_name)
        detours = rng.sample(temp_task_names, 3)
        detours_instructions = []
        for task_name in detours:
            task_func, task_kwargs, transform_func, transform_kwargs, detour_reasoning = self.ARC_1D_TASKS[task_name]
            detour_instruction = template1.format(instruction=detour_reasoning)
            for i, example in enumerate(train_examples, 1):
                try:
                    transformed_detour = transform_func(example["input"], **transform_kwargs)
                    detour_instruction += (f"Example {i}: " + " ".join(str(x) for x in transformed_detour) + "\n")
                except Exception as e:
                    detour_instruction += (f"Example {i}: " + "the described transformation is impossible to be applied to this example" + "\n")
            detour_instruction += template3
            detours_instructions.append(detour_instruction)

        ### format trajectories
        optimal_trajectory = "Looking at these three examples. " + optimal
        
        backtrack_temp = "Looking at these three examples. "

        for i, detour_instruction in enumerate(detours_instructions):
            backtrack_temp += detour_instruction
            if i == 0:
                one_backtrack = backtrack_temp + optimal
            elif i == 1:
                two_backtrack = backtrack_temp + optimal
            elif i == 2:
                three_backtrack = backtrack_temp + optimal
        



        # Format question
        question = "Find the common rule that maps an input grid to an output grid, given the examples below.\n\n"

        # Add training examples
        for i, example in enumerate(train_examples, 1):
            question += f"Example {i}:\n"
            question += "Input:  " + " ".join(str(x) for x in example["input"]) + "\n"
            question += "Output: " + " ".join(str(x) for x in example["output"]) + "\n\n"

        # Add test input
        question += "Below is a test input grid. Predict the corresponding output grid by applying the rule you found.\n"
        # question += "Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. "
        # question += "Your final answer must be placed in <output></output> tags and should be just be the text output grid itself.\n\n"
        question += "Input:\n"
        question += " ".join(str(x) for x in test_example["input"])

        return {
            "question": question,
            "answer": " ".join(str(x) for x in test_example["output"]),
            "metadata": {
                "task_name": gold_task_name,
                "size": size,
                "train_examples": train_examples,
                "test_example": test_example,
                "optimal_trajectory": optimal_trajectory,
                "one_backtrack": one_backtrack,
                "two_backtrack": two_backtrack,
                "three_backtrack": three_backtrack,
            },
        }


# Register the dataset
register_dataset("arc_1d", Arc1DDataset, Arc1DConfig)
