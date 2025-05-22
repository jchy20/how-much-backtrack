import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import sympy
from sympy.geometry import Point

from ..factory import ProceduralDataset, register_dataset


@dataclass
class AdvancedGeometryConfig:
    """
    Configuration for generating advanced geometry tasks.
    """

    min_coord: int = -10  # Minimum x/y coordinate
    max_coord: int = 10  # Maximum x/y coordinate
    size: int = 50  # Number of problems to generate
    seed: Optional[int] = None

    # Probability or list of tasks we want to generate
    # For demonstration, we have three categories:
    task_types: list[str] = field(
        default_factory=lambda: [
            "orthocenter",
            "incircle_radius",
            "angle_measure",
        ]
    )

    def validate(self):
        assert self.min_coord < self.max_coord, "min_coord must be < max_coord."
        assert self.size > 0, "Size of dataset must be positive."
        assert len(self.task_types) > 0, "Must specify at least one task type."


# Join format instructions into a single string
GEOMETRY_FORMAT_INSTRUCTIONS = "\n".join(
    [
        "For all geometry problems:",
        "1. Give coordinates in the form (x, y)",
        "2. Round decimal answers to 3 decimal places",
        "3. Use the degree symbol ° for angles",
        "4. Return only th angle, coordinates, or radius as your answer.",
    ]
)


class AdvancedGeometryDataset(ProceduralDataset):
    """
    A dataset for advanced geometry tasks using coordinate geometry.
    """

    def __init__(self, config: AdvancedGeometryConfig):
        # self._prompt_templates = {
        #     "orthocenter": [
        #         f"Given triangle ABC with coordinates A={{A}}, B={{B}}, and C={{C}}, find the coordinates of its orthocenter. {GEOMETRY_FORMAT_INSTRUCTIONS}",
        #         f"For triangle with vertices A={{A}}, B={{B}}, and C={{C}}, determine the orthocenter (intersection of altitudes). {GEOMETRY_FORMAT_INSTRUCTIONS}",
        #     ],
        #     "incircle_radius": [
        #         f"Consider triangle ABC with coordinates A={{A}}, B={{B}}, and C={{C}}. Compute the radius of its incircle. {GEOMETRY_FORMAT_INSTRUCTIONS}",
        #         f"Find the incircle radius of triangle ABC whose vertices are A={{A}}, B={{B}}, and C={{C}}. {GEOMETRY_FORMAT_INSTRUCTIONS}",
        #     ],
        #     "angle_measure": [
        #         f"In triangle ABC with coordinates A={{A}}, B={{B}}, and C={{C}}, find the measure (in degrees) of angle ABC. {GEOMETRY_FORMAT_INSTRUCTIONS}",
        #         f"Given a triangle with vertices A={{A}}, B={{B}}, and C={{C}}, determine the angle at B in degrees. {GEOMETRY_FORMAT_INSTRUCTIONS}",
        #     ],
        # }
        self._prompt_templates = {
            "orthocenter": [
                f"Given triangle ABC with coordinates A={{A}}, B={{B}}, and C={{C}}, find the coordinates of its orthocenter.",
                f"For triangle with vertices A={{A}}, B={{B}}, and C={{C}}, determine the orthocenter (intersection of altitudes).",
            ],
            "incircle_radius": [
                f"Consider triangle ABC with coordinates A={{A}}, B={{B}}, and C={{C}}. Compute the radius of its incircle.",
                f"Find the incircle radius of triangle ABC whose vertices are A={{A}}, B={{B}}, and C={{C}}.",
            ],
            "angle_measure": [
                f"In triangle ABC with coordinates A={{A}}, B={{B}}, and C={{C}}, find the measure (in degrees) of angle ABC.",
                f"Given a triangle with vertices A={{A}}, B={{B}}, and C={{C}}, determine the angle at B in degrees.",
            ],
        }
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single advanced geometry item based on the config's task types.
        """
        rng = random.Random(self.seed + idx)
        task_type = rng.choice(self.config.task_types)

        # Randomly generate coordinates for a triangle
        A, B, C = self._generate_non_degenerate_triangle(rng)

        # Build a question and compute the solution
        if task_type == "orthocenter":
            question, answer, metadata = self._build_orthocenter_task(rng, A, B, C)
        elif task_type == "incircle_radius":
            question, answer, metadata = self._build_incircle_radius_task(rng, A, B, C)
        elif task_type == "angle_measure":
            question, answer, metadata = self._build_angle_measure_task(rng, A, B, C)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        metadata["task_type"] = task_type

        return {
            "question": question,
            "answer": answer,
            "metadata": metadata,
        }

    def _generate_non_degenerate_triangle(self, rng: random.Random):
        """
        Generate a random non-degenerate triangle with integer coordinates
        in [min_coord, max_coord] x [min_coord, max_coord].
        """
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate points with integer coordinates
            points = []
            for _ in range(3):
                x = rng.randint(self.config.min_coord, self.config.max_coord)
                y = rng.randint(self.config.min_coord, self.config.max_coord)
                points.append(Point(x, y))

            A, B, C = points

            # Calculate signed area to check for non-degeneracy
            # Using the formula: 1/2 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
            area = abs(A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2

            if area > 0:
                return A, B, C

        raise ValueError(f"Failed to generate a non-degenerate triangle after {max_attempts} attempts.")

    def _build_orthocenter_task(self, rng: random.Random, A: Point, B: Point, C: Point):
        """
        Build a question about finding the orthocenter of triangle ABC.
        """
        # Optimal reasoning trajectories (we build errors into this)
        reasoning_steps = []
        reasoning_steps.append(f"Let's first create the lines of the triangle. First, let's create a line from point B({B.x}, {B.y}) to point C({C.x}, {C.y}).")
        reasoning_steps.append(f"Next, create a line from point C({C.x}, {C.y}) to point A({A.x}, {A.y}).")
        reasoning_steps.append(f"Now, let's find the altitudes of the triangle. First, find the altitude from point A({A.x}, {A.y}) by creating a line perpendicular to BC passing through A.")
        reasoning_steps.append(f"Next, find the altitude from point B({B.x}, {B.y}) by creating a line perpendicular to CA passing through B.")
        reasoning_steps.append(f"The orthocenter is the intersection point of the two altitudes we found.")
        reasoning_steps.append(f"Now, let's calculate the exact coordinates of the intersection point.")

        def construct_answers(original_A: Point, original_B: Point, original_C: Point, error: bool = False) -> tuple[Point, str]:
            A = original_A
            B = original_B
            C = original_C
            random_gen = random.Random()
            if error:
                random_step = random_gen.randint(0, 4)
            else:
                random_step = -1

            # Convert segments to lines
            if random_step == 0:
                B = B + Point(random_gen.randint(-10, 10), random_gen.randint(-10, 10))
            if random_step == 1:
                C = C + Point(random_gen.randint(-10, 10), random_gen.randint(-10, 10))
            if random_step == 2:
                A = A + Point(random_gen.randint(-10, 10), random_gen.randint(-10, 10))
            BC_line = sympy.Line(B, C)
            CA_line = sympy.Line(C, A)

            # Calculate altitudes by creating lines perpendicular from each vertex
            if random_step == 3:
                A = A + Point(random_gen.randint(-10, 10), random_gen.randint(-10, 10))
            if random_step == 4:
                B = B + Point(random_gen.randint(-10, 10), random_gen.randint(-10, 10))
            alt_A = BC_line.perpendicular_line(A)
            alt_B = CA_line.perpendicular_line(B)

            # Find orthocenter (intersection of any two altitudes, e.g. alt_A and alt_B)
            intersection_points = alt_A.intersection(alt_B)
            if not intersection_points or not isinstance(intersection_points[0], sympy.geometry.Point):
                raise ValueError("Altitudes don't meet at a single point")
            
            ortho = intersection_points[0]

            x_ortho_approx = float(ortho.x.evalf())
            y_ortho_approx = float(ortho.y.evalf())
            ortho_str = f"({x_ortho_approx:.3f}, {y_ortho_approx:.3f})"

            reasoning_steps = []
            reasoning_steps.append(f"I have obtained the answer {ortho_str}. Let's double check it.")
            reasoning_steps.append(f"Let's first check if I have calculated the line from point B({original_B.x}, {original_B.y}) to point C({original_C.x}, {original_C.y}) correctly.")
            reasoning_steps.append(f"Let's check if I have calculated the line from point C({original_C.x}, {original_C.y}) to point A({original_A.x}, {original_A.y}) correctly.")
            reasoning_steps.append(f"Let's check if I have calculated the altitude from point A({original_A.x}, {original_A.y}) correctly.")
            reasoning_steps.append(f"Let's check if I have calculated the altitude from point B({original_B.x}, {original_B.y}) correctly.")

            if random_step == -1:
                reasoning = reasoning_steps[0] + "\n"
                for step in reasoning_steps[1:]:
                    reasoning += step + " I think this step is done correctly." + "\n"
                reasoning += "Lastly, let's check if I have calculated the intersection of the two altitudes correctly. And I think this step is done correctly, so the answer is correct"
            elif random_step == 0:
                reasoning = reasoning_steps[0] + "\n" + reasoning_steps[1]
                reasoning += f" Oh, I made a mistake here, and {ortho_str} is not the correct answer. Let's try to recalculate the line from B to C, and obtain a new answer following the same steps as before."
            elif random_step == 1:
                reasoning = reasoning_steps[0] + "\n" + reasoning_steps[1] + " I think this step is done correctly." + "\n" + reasoning_steps[2]
                reasoning += f" Oh, I made a mistake here, and {ortho_str} is not the correct answer. Let's try to recalculate the line from C to A, and obtain a new answer following the same steps as before."
            elif random_step == 2:
                reasoning = reasoning_steps[0] + "\n" + reasoning_steps[1] + " Wait, I have made a mistake calculating the line from B to C." + "\n" + reasoning_steps[2] + " I have also made a mistake calculating the line from C to A. So {ortho_str} is not the correct answer."
                reasoning += f"Let's try to recalculate the line from B to C, and the line from C to A, and obtain a new answer following the same steps as before."
            elif random_step == 3:
                reasoning = reasoning_steps[0] + "\n"
                for step in reasoning_steps[1:3]:
                    reasoning += step + " I think this step is done correctly." + "\n"
                reasoning += reasoning_steps[3] + " Oh, I made a mistake here, and {ortho_str} is not the correct answer. Let's try to recalculate the altitude from A, and obtain a new answer following the same steps as before."
            elif random_step == 4:
                reasoning = reasoning_steps[0] + "\n"
                for step in reasoning_steps[1:4]:
                    reasoning += step + " I think this step is done correctly." + "\n"
                reasoning += reasoning_steps[4] + " Oh, I made a mistake here, and {ortho_str} is not the correct answer. Let's try to recalculate the altitude from B, and obtain a new answer following the same steps as before."
            return ortho, reasoning, random_step
        
        reasoning_prefix = "\n".join(reasoning_steps) + "\n\n"
        
        # Keep trying until we get a valid orthocenter
        while True:
            try:
                ortho, optimal_reasoning, _ = construct_answers(A, B, C, error = False)
                break
            except ValueError:
                # Regenerate triangle points
                A, B, C = self._generate_non_degenerate_triangle(rng)
                continue

        x_ortho_approx = float(ortho.x.evalf())
        y_ortho_approx = float(ortho.y.evalf())

        question_template = rng.choice(self._prompt_templates["orthocenter"])
        question = question_template.format(A=(A.x, A.y), B=(B.x, B.y), C=(C.x, C.y), a="a", b="b")
        answer_str = f"({x_ortho_approx:.3f}, {y_ortho_approx:.3f})"

        optimal_reasoning = optimal_reasoning + "\n" + "</think>\n\n" + "<answer>\n" + answer_str + "\n</answer>"

        detours_dict = {}
        for i in range(3):
            while True:
                try:
                    detour_ortho, detour_reasoning, detour_step = construct_answers(A, B, C, error = True)
                    detours_dict[i] = (detour_ortho, detour_reasoning, detour_step)
                    break
                except ValueError:
                    # Regenerate triangle points
                    A, B, C = self._generate_non_degenerate_triangle(rng)
                    continue

        # Sort detours_dict based on detour_step from small to large
        sorted_detours = dict(sorted(detours_dict.items(), key=lambda item: item[1][2]))
        detours_dict = sorted_detours

        reasoning_chain = reasoning_prefix
        for i, (key, value) in enumerate(detours_dict.items()):
            detour_ortho, detour_reasoning, detour_step = value
            reasoning_chain += detour_reasoning + "\n\n"
            if i == 0:
                one_backback = reasoning_chain + optimal_reasoning
            if i == 1:
                two_backtrack = reasoning_chain + optimal_reasoning
            if i == 2:
                three_backtrack = reasoning_chain + optimal_reasoning

        optimal = reasoning_prefix + optimal_reasoning
        
        metadata = {
            "A": (A.x, A.y),
            "B": (B.x, B.y),
            "C": (C.x, C.y),
            "ortho": (ortho.x, ortho.y),
            "orthocenter_exact": (str(ortho.x), str(ortho.y)),
            "orthocenter_approx": (x_ortho_approx, y_ortho_approx),
            "optimal_reasoning": optimal,
            "one_backtrack": one_backback,
            "two_backtrack": two_backtrack,
            "three_backtrack": three_backtrack,
        }
        return question, answer_str, metadata

    def _build_incircle_radius_task(self, rng: random.Random, A: Point, B: Point, C: Point):
        """
        Build a question about finding the incircle radius of triangle ABC.
        """
        # Calculate side lengths
        def calculate_side_lengths(A: Point, B: Point, C: Point, random_step: int = -1):
            random_gen = random.Random()
            if random_step == 0:
                B = B + Point(random_gen.randint(-10, 10), random_gen.randint(-10, 10))
            elif random_step == 1:
                C = C + Point(random_gen.randint(-10, 10), random_gen.randint(-10, 10))
            elif random_step == 2:
                A = A + Point(random_gen.randint(-10, 10), random_gen.randint(-10, 10))

            a = B.distance(C)
            b = C.distance(A)
            c = A.distance(B)
            step = f"First, compute the three side lengths of the triangle. We have a = {float(a.evalf()):.3f}, b = {float(b.evalf()):.3f}, and c = {float(c.evalf()):.3f}."
            return a, b, c, step
            
        # Semi-perimeter
        def calculate_semi_perimeter(a: float, b: float, c: float, error: bool = False):
            random_gen = random.Random()
            if error:
                temp_a = a + random_gen.randint(1, 20) / 10
                s = (temp_a + b + c) / 2
            else:
                s = (a + b + c) / 2
            step = f"Next, find the semiperimeter s = ({float(a.evalf()):.3f} + {float(b.evalf()):.3f} + {float(c.evalf()):.3f}) / 2. We have s = {float(s.evalf()):.3f}."
            return s, step

        # Area using Heron's formula
        def calculate_area(s: float, a: float, b: float, c: float, error: bool = False):
            random_gen = random.Random()
            if error:
                temp_s = s + random_gen.randint(1, 20) / 10
                area = sympy.sqrt(temp_s * (temp_s - a) * (temp_s - b) * (temp_s - c))
            else:
                area = sympy.sqrt(s * (s - a) * (s - b) * (s - c))
            step = f"Use Heron's formula to compute the area: Δ = √[{float(s.evalf()):.3f} * ({float(s.evalf()):.3f} − {float(a.evalf()):.3f}) * ({float(s.evalf()):.3f} − {float(b.evalf()):.3f}) * ({float(s.evalf()):.3f} − {float(c.evalf()):.3f})]. We have Δ = {float(area.evalf()):.3f}."
            return area, step

        # Radius of incircle = Area / Semi-perimeter
        def calculate_in_circle_radius(area: float, s: float):
            radius = area / s
            step = f"The in‑circle radius is r = {float(area.evalf()):.3f} / {float(s.evalf()):.3f}. We have r = {float(radius.evalf()):.3f}."
            return radius, step


        def build_answer(A: Point, B: Point, C: Point, error: bool = False):
            correction_reasonings = []
            correction_reasonings.append("I have obtained the answer {radius_str}. Let's double check it.")
            correction_reasonings.append(f"Let's first check if I have calculated the side lengths of the triangle correctly.")
            correction_reasonings.append(f"Let's check if I have calculated the semiperimeter correctly.")
            correction_reasonings.append(f"Let's check if I have calculated the area correctly.")
            correction_reasonings.append(f"Let's check if I have calculated the in-circle radius correctly.")


            if error:
                random_gen = random.Random()
                random_step = random_gen.randint(0, 4)
            else:
                random_step = -1
                a, b, c, step1 = calculate_side_lengths(A, B, C, random_step)
                s, step2 = calculate_semi_perimeter(a, b, c, error = False)
                area, step3 = calculate_area(s, a, b, c, error = False)
                radius, step4 = calculate_in_circle_radius(area, s)
                radius_approx = float(radius.evalf())
                correction_reasoning = correction_reasonings[0].format(radius_str = f"{radius_approx:.3f}") + "\n" + correction_reasonings[1] + f" {float(a.evalf()):.3f}, {float(b.evalf()):.3f}, {float(c.evalf()):.3f} all look corrrect" + "\n" 
                correction_reasoning += correction_reasonings[2] + f" {float(s.evalf()):.3f} looks correct" + "\n" + correction_reasonings[3] + f" {float(area.evalf()):.3f} looks correct" + "\n" + correction_reasonings[4] + f" {radius_approx:.3f} looks correct"
            
            if random_step == 0:
                a, b, c, step1 = calculate_side_lengths(A, B, C, random_step)
                s, step2 = calculate_semi_perimeter(a, b, c, error = False)
                area, step3 = calculate_area(s, a, b, c, error = False)
                radius, step4 = calculate_in_circle_radius(area, s)
                radius_approx = float(radius.evalf())
                correction_reasoning = correction_reasonings[0].format(radius_str = f"{radius_approx:.3f}") + "\n" + correction_reasonings[1]
                correction_reasoning += f" Oh, I made a mistake here, I have calculated the side lengths from A to C and side lengths from A to B incorrectly, which leads to {radius_approx:.3f} being the wrong answer. Let's try to recalculate the side lengths, and obtain a new answer."
            
            if random_step == 1:
                a, b, c, step1 = calculate_side_lengths(A, B, C, random_step)
                s, step2 = calculate_semi_perimeter(a, b, c, error = False)
                area, step3 = calculate_area(s, a, b, c, error = False)
                radius, step4 = calculate_in_circle_radius(area, s)
                radius_approx = float(radius.evalf())
                correction_reasoning = correction_reasonings[0].format(radius_str = f"{radius_approx:.3f}") + "\n" + correction_reasonings[1]
                correction_reasoning += f" Oh, I made a mistake here, I have calculated the side lengths from B to C and side lengths from A to B incorrectly, which leads to {radius_approx:.3f} being the wrong answer. Let's try to recalculate the side lengths, and obtain a new answer."
            
            if random_step == 2:
                a, b, c, step1 = calculate_side_lengths(A, B, C, random_step)
                s, step2 = calculate_semi_perimeter(a, b, c, error = False)
                area, step3 = calculate_area(s, a, b, c, error = False)
                radius, step4 = calculate_in_circle_radius(area, s)
                radius_approx = float(radius.evalf())
                correction_reasoning = correction_reasonings[0].format(radius_str = f"{radius_approx:.3f}") + "\n" + correction_reasonings[1]
                correction_reasoning += f" Oh, I made a mistake here, I have calculated the side lengths from A to C and side lengths from B to C incorrectly, which leads to {radius_approx:.3f} being the wrong answer. Let's try to recalculate the side lengths, and obtain a new answer."

            if random_step == 3:
                a, b, c, step1 = calculate_side_lengths(A, B, C, random_step)
                s, step2 = calculate_semi_perimeter(a, b, c, error = True)
                area, step3 = calculate_area(s, a, b, c, error = False)
                radius, step4 = calculate_in_circle_radius(area, s)
                radius_approx = float(radius.evalf())
                correction_reasoning = correction_reasonings[0].format(radius_str = f"{radius_approx:.3f}") + "\n" + correction_reasonings[1] + f" {float(a.evalf()):.3f}, {float(b.evalf()):.3f}, {float(c.evalf()):.3f} all look corrrect" + "\n" + correction_reasonings[2]
                correction_reasoning += f" Oh, I made a mistake here, I have calculated the semiperimeter incorrectly, which leads to {radius_approx:.3f} being the wrong answer. Let's try to recalculate the semiperimeter, and obtain a new answer."

            if random_step == 4:
                a, b, c, step1 = calculate_side_lengths(A, B, C, random_step)
                s, step2 = calculate_semi_perimeter(a, b, c, error = False)
                area, step3 = calculate_area(s, a, b, c, error = True)
                radius, step4 = calculate_in_circle_radius(area, s)
                radius_approx = float(radius.evalf())
                correction_reasoning = correction_reasonings[0].format(radius_str = f"{radius_approx:.3f}") + "\n" + correction_reasonings[1] + f" {float(a.evalf()):.3f}, {float(b.evalf()):.3f}, {float(c.evalf()):.3f} all look corrrect" + "\n" + correction_reasonings[2] + f" {float(s.evalf()):.3f} looks correct" + "\n" + correction_reasonings[3]
                correction_reasoning += f" Oh, I made a mistake here, I have calculated the area incorrectly, which leads to {radius_approx:.3f} being the wrong answer. Let's try to recalculate the area, and obtain a new answer."
            
            reasoning_steps = [step1, step2, step3, step4]
            reasoning_chain = "\n".join(reasoning_steps) + "\n\n" + correction_reasoning + "\n\n"

            return radius, radius_approx, reasoning_chain, random_step

        optimal_radius, optimal_radius_approx, optimal_reasoning, _ = build_answer(A, B, C, error = False)
        optimal_reasoning = optimal_reasoning + "\n" + "</think>\n\n" + "<answer>\n" + f"{optimal_radius_approx:.3f}" + "\n</answer>"

        detours_dict = {}
        for i in range(3):
            detour_radius, detour_radius_approx, detour_reasoning, detour_step = build_answer(A, B, C, error = True)
            detours_dict[i] = (detour_radius, detour_radius_approx, detour_reasoning, detour_step)
        detours_dict = dict(sorted(detours_dict.items(), key=lambda item: item[1][3]))

        reasoning_chain = ""
        for i, (key, value) in enumerate(detours_dict.items()):
            detour_radius, detour_radius_approx, detour_reasoning, detour_step = value
            reasoning_chain += detour_reasoning + "\n\n"
            if i == 0:
                one_backback = reasoning_chain + optimal_reasoning
            if i == 1:
                two_backtrack = reasoning_chain + optimal_reasoning
            if i == 2:
                three_backtrack = reasoning_chain + optimal_reasoning




        question_template = rng.choice(self._prompt_templates["incircle_radius"])
        question = question_template.format(A=(A.x, A.y), B=(B.x, B.y), C=(C.x, C.y))
        answer_str = f"{optimal_radius_approx:.3f}"

        metadata = {
            "A": (A.x, A.y),
            "B": (B.x, B.y),
            "C": (C.x, C.y),
            "incircle_radius_exact": str(optimal_radius),
            "incircle_radius_approx": optimal_radius_approx,
            "optimal_reasoning": optimal_reasoning,
            "one_backtrack": one_backback,
            "two_backtrack": two_backtrack,
            "three_backtrack": three_backtrack,
        }
        return question, answer_str, metadata


    # def _build_angle_measure_task(self, rng: random.Random, A: Point, B: Point, C: Point):
    #     """
    #     Build a question about finding the measure of angle ABC in degrees.
    #     """
    #     # Angle at B means the angle ∠ABC
    #     # Vector BA = A - B, BC = C - B
    #     BA = A - B
    #     BC = C - B

    #     # Use vector dot product to find angle between BA and BC
    #     # angle = arccos((BA · BC) / (|BA| * |BC|))
    #     dot_val = BA.dot(BC)
    #     mag_ba = BA.distance(Point(0, 0))
    #     mag_bc = BC.distance(Point(0, 0))

    #     # numerical check
    #     if mag_ba == 0 or mag_bc == 0:
    #         # degenerate, but theoretically we forced a non-degenerate triangle
    #         angle_deg = 0
    #     else:
    #         cos_theta = dot_val / (mag_ba * mag_bc)
    #         # clamp cos_theta to [-1, 1] to avoid floating rounding errors
    #         cos_theta = max(-1, min(1, cos_theta))
    #         angle_rad = sympy.acos(cos_theta)
    #         angle_deg = float(angle_rad.evalf() * 180 / sympy.pi)

    #     question_template = rng.choice(self._prompt_templates["angle_measure"])
    #     question = question_template.format(A=(A.x, A.y), B=(B.x, B.y), C=(C.x, C.y), a="a", b="b")

    #     answer_str = f"{angle_deg:.2f}°"
    #     metadata = {
    #         "A": (A.x, A.y),
    #         "B": (B.x, B.y),
    #         "C": (C.x, C.y),
    #         "angle_ABC_degrees": angle_deg,
    #     }
    #     return question, answer_str, metadata

    def _build_angle_measure_task(self, rng: random.Random, A: Point, B: Point, C: Point):
        """
        Build a question about measuring angle ∠ABC (at vertex B) in degrees,
        mirroring the detour / optimal–back‑tracking structure of
        `_build_orthocenter_task`.
        """

        # ------------------ canonical reasoning skeleton ----------------------- #
        base_steps = [
            f"Let's first compute the vectors that form the angle at B. Vector BA is A({A.x}, {A.y}) minus B({B.x}, {B.y}).",
            f"Vector BC is C({C.x}, {C.y}) minus B({B.x}, {B.y}).",
            "Now, we'll take the dot product of BA and BC.",
            "Next, we'll compute the magnitudes |BA| and |BC|.",
            "Finally, we use the formula θ = arccos( dot / (|BA|·|BC|) ) and convert to degrees.",
        ]

        # ------------------ internal helper ----------------------------------- #
        def construct_answers(orig_A: Point, orig_B: Point, orig_C: Point, *, error: bool = False):
            """Return (angle_deg, reasoning_text, perturbation_id)."""
            A, B, C = orig_A, orig_B, orig_C
            rng_local = random.Random()
            step = rng_local.randint(0, 4) if error else -1  # which step to corrupt

            # Step‑level corruption by perturbing one of the points, mirroring orthocenter logic
            if step == 0:
                A = A + Point(rng_local.randint(-10, 10), rng_local.randint(-10, 10))
            elif step == 1:
                C = C + Point(rng_local.randint(-10, 10), rng_local.randint(-10, 10))
            elif step == 2:
                B = B + Point(rng_local.randint(-10, 10), rng_local.randint(-10, 10))
            # Steps 3 & 4 keep points but we will mis‑evaluate either magnitude or arccos below

            # ---- compute canonical quantities ----
            BA = A - B
            BC = C - B
            dot_val = BA.dot(BC)
            mag_ba = BA.distance(Point(0, 0))
            mag_bc = BC.distance(Point(0, 0))

            # Handle degenerate
            if mag_ba == 0 or mag_bc == 0:
                theta_deg = 0.0
            else:
                cos_theta = dot_val / (mag_ba * mag_bc)
                cos_theta = max(-1, min(1, cos_theta))

                if step == 3 and error:          # inject error: forget to clamp → possible domain error
                    # purposely overflow slightly
                    cos_theta += 0.2
                if not (-1 <= cos_theta <= 1):
                    # fallback: cap but note the error
                    cos_theta = 1.0 if cos_theta > 1 else -1.0

                theta_rad = sympy.acos(cos_theta)
                theta_deg = float(theta_rad.evalf() * 180 / sympy.pi)

                if step == 4 and error:
                    theta_deg += rng_local.uniform(5.0, 25.0)  # arithmetic slip

            angle_str = f"{theta_deg:.2f}°"

            # ---------------- build self‑check reasoning ----------------------- #
            review_steps = [
                f"I have obtained the answer {angle_str}. Let's double check it.",
                f"Let's verify that vector BA was computed correctly from A({orig_A.x}, {orig_A.y}) and B({orig_B.x}, {orig_B.y}).",
                f"Let's verify that vector BC was computed correctly from B({orig_B.x}, {orig_B.y}) and C({orig_C.x}, {orig_C.y}).",
                "Let's verify the dot product calculation.",
                "Let's verify the magnitude calculations and the arccos step.",
            ]

            if step == -1:
                reasoning = review_steps[0] + "\n"
                for s in review_steps[1:]:
                    reasoning += s + " I think this step is done correctly.\n"
                reasoning += "Everything checks out, so the answer is correct."
            else:
                # mark the earliest incorrect step
                reasoning = "\n".join(review_steps[: step + 2])
                reasoning += f" Oh, I made a mistake here, and {angle_str} is not the correct answer. Let's recompute this step and follow through the same pipeline."

            return theta_deg, reasoning, step

        # ---------------- produce optimal + detours --------------------------- #
        prefix = "\n".join(base_steps) + "\n\n"

        angle_opt, reasoning_opt, _ = construct_answers(A, B, C, error=False)
        angle_str_opt = f"{angle_opt:.2f}°"

        reasoning_opt = reasoning_opt + "\n" + "</think>\n\n<answer>\n" + angle_str_opt + "\n</answer>"

        detours = {}
        for i in range(3):
            det_angle, det_reason, det_step = construct_answers(A, B, C, error=True)
            detours[i] = (det_angle, det_reason, det_step)

        detours = dict(sorted(detours.items(), key=lambda kv: kv[1][2]))

        chain = prefix
        one_backtrack = two_backtrack = three_backtrack = None
        for i, (_, (da, dr, ds)) in enumerate(detours.items()):
            chain += dr + "\n\n"
            if i == 0:
                one_backtrack = chain + reasoning_opt
            elif i == 1:
                two_backtrack = chain + reasoning_opt
            elif i == 2:
                three_backtrack = chain + reasoning_opt

        optimal_chain = prefix + reasoning_opt

        # ---------------- craft question & metadata --------------------------- #
        q_template = rng.choice(self._prompt_templates["angle_measure"])
        question = q_template.format(A=(A.x, A.y), B=(B.x, B.y), C=(C.x, C.y), a="a", b="b")

        answer_str = angle_str_opt

        metadata = {
            "A": (A.x, A.y),
            "B": (B.x, B.y),
            "C": (C.x, C.y),
            "angle_ABC_degrees": angle_opt,
            "optimal_reasoning": optimal_chain,
            "one_backtrack": one_backtrack,
            "two_backtrack": two_backtrack,
            "three_backtrack": three_backtrack,
        }
        return question, answer_str, metadata


    def score_answer(self, answer: str | None, entry: dict[str, Any]) -> float:
        reward = 0.0
        expected_answer = entry["answer"]
        metadata = entry["metadata"]
        task_type = metadata["task_type"]

        if answer is not None:
            try:
                if metadata["task_type"] == "angle_measure":
                    answer = answer.replace("\u00b0", "")
                    expected_answer = expected_answer.replace("\u00b0", "")
                    if np.round(float(answer), 2) == np.round(float(expected_answer), 2):
                        reward = 1.0
                    elif len(answer.strip()) > 0:
                        reward = 0.05
                    else:
                        reward = 0.01
                elif metadata["task_type"] == "orthocenter":
                    x_coord = answer.split(",")[0].replace("(", "").strip()
                    y_coord = answer.split(",")[1].replace(")", "").strip()

                    expected_x = metadata["ortho"][0]
                    expected_y = metadata["ortho"][1]

                    if x_coord == expected_x and y_coord == expected_y:
                        reward = 1.0
                    elif (np.round(float(x_coord), 2) == np.round(float(expected_x), 2)) and (
                        np.round(float(y_coord), 2) == np.round(float(expected_y), 2)
                    ):
                        reward = 1.0
                    elif len(x_coord.strip()) > 0 and len(y_coord.strip()) > 0:
                        reward = 0.05
                    else:
                        reward = 0.01
                elif metadata["task_type"] == "incircle_radius":
                    if answer == expected_answer:
                        reward = 1.0
                    elif np.round(float(answer), 2) == np.round(float(metadata["incircle_radius_exact"]), 2):
                        reward = 0.5
                    elif len(answer.strip()) > 0:
                        reward = 0.05
                    else:
                        reward = 0.01
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")
            except:
                reward = 0.01
        return reward


# Register the dataset
register_dataset("advanced_geometry", AdvancedGeometryDataset, AdvancedGeometryConfig)
