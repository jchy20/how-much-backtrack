from random import Random
from typing import Optional


def gen_field(size: int, color: int = 0) -> list[int]:
    """Generate a field of given size filled with specified color (default 0)."""
    return [color] * size


def write_block(pos: int, block: list[int], field: list[int]) -> list[int]:
    """Write a block into a field at given position."""
    result = field.copy()
    for i, color in enumerate(block):
        result[pos + i] = color
    return result


def task_move_n_pix(rng: Random, size: int, move_pix: int, solid: bool) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block is moved to the right by move_pix pixels."""
    if size <= move_pix + 1:
        return None

    block_size = rng.randint(1, size - move_pix - 1)
    block_pos = rng.randint(0, size - block_size - move_pix)

    if solid:
        color = rng.randint(1, 9)
        block = [color] * block_size
    else:
        block = [rng.randint(1, 9) for _ in range(block_size)]

    question = write_block(block_pos, block, gen_field(size))
    answer = write_block(block_pos + move_pix, block, gen_field(size))

    return {"input": question, "output": answer}



def transform_move_n_pix(input_grid: list[int], move_pix: int, direction: str = "right") -> list[int]:
    size = len(input_grid)
    output = [0] * size  # Initialize output with zeros
    
    # For each position in input
    for i in range(size):
        if input_grid[i] != 0:  # If there's a pixel
            if direction == "right":
                new_pos = i + move_pix
            else:  # left
                new_pos = i - move_pix
            if 0 <= new_pos < size:  # Only move if within bounds
                output[new_pos] = input_grid[i]
            
    return output
    
    
def task_move_n_pix_wrapped(rng: Random, size: int, move_pix: int, solid: bool) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block is moved to the right by move_pix pixels with wrapping."""
    block_size = rng.randint(1, size)
    block_pos = rng.randint(0, size - 1)

    if solid:
        color = rng.randint(1, 9)
        block = [color] * block_size
    else:
        block = [rng.randint(1, 9) for _ in range(block_size)]

    question = gen_field(size)
    answer = gen_field(size)

    for i, color in enumerate(block):
        question[(block_pos + i) % size] = color
        answer[(block_pos + move_pix + i) % size] = color

    return {"input": question, "output": answer}


def transform_move_n_pix_wrapped(input_grid: list[int], move_pix: int, direction: str = "right") -> list[int]:
    size = len(input_grid)
    output = [0] * size  # Initialize output with zeros
    
    # For each position in input
    for i in range(size):
        if input_grid[i] != 0:  # If there's a pixel
            # Calculate new position with wrapping
            if direction == "right":
                new_pos = (i + move_pix) % size
            else:  # left
                new_pos = (i - move_pix) % size
            output[new_pos] = input_grid[i]
            
    return output


def task_gravity(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where all non-zero elements are attracted to the left."""
    density = 0.5
    question = [rng.randint(1, 9) if rng.random() < density else 0 for _ in range(size)]

    non_zero = [x for x in question if x != 0]
    answer = non_zero + [0] * (size - len(non_zero))

    return {"input": question, "output": answer}

def transform_gravity(input_grid: list[int], direction: str = "right") -> list[int]:
    # Extract all non-zero elements
    non_zero = [x for x in input_grid if x != 0]
    size = len(input_grid)
    
    if direction == "right":
        # Place non-zero elements at the start
        return non_zero + [0] * (size - len(non_zero))
    else:
        # Place non-zero elements at the end
        return [0] * (size - len(non_zero)) + non_zero


def task_gravity_counting(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where non-zero elements are counted and represented as a sequence of 1s."""
    density = 0.5
    question = [rng.randint(1, 9) if rng.random() < density else 0 for _ in range(size)]

    count = sum(1 for x in question if x != 0)
    answer = [1] * count + [0] * (size - count)

    return {"input": question, "output": answer}

def transform_gravity_counting(input_grid: list[int], direction: str = "right") -> list[int]:
    # Count non-zero elements
    count = sum(1 for x in input_grid if x != 0)
    size = len(input_grid)
    
    if direction == "right":
        # Place 1s at the start
        return [1] * count + [0] * (size - count)
    else:
        # Place 1s at the end
        return [0] * (size - count) + [1] * count


def task_gravity_antigravity(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where color 1 moves right and color 2 moves left."""
    density = 0.5
    question = [rng.randint(1, 2) if rng.random() < density else 0 for _ in range(size)]

    color1 = [x for x in question if x == 1]
    color2 = [x for x in question if x == 2]
    answer = [2] * len(color2) + [0] * (size - len(color1) - len(color2)) + [1] * len(color1)

    return {"input": question, "output": answer}


def transform_gravity_antigravity(input_grid: list[int], direction: str = "right") -> list[int]:
    size = len(input_grid)
    
    # Extract colors
    color1 = [x for x in input_grid if x == 1]  # Get all 1s
    color2 = [x for x in input_grid if x == 2]  # Get all 2s
    
    if direction == "right":
        # color 2s on left, color 1s on right
        return [2] * len(color2) + [0] * (size - len(color1) - len(color2)) + [1] * len(color1)
    else:
        # color 1s on left, color 2s on right
        return [1] * len(color1) + [0] * (size - len(color1) - len(color2)) + [2] * len(color2)


def task_block_touch_dot(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block moves to touch (but not cover) a dot."""
    dot_color = 1
    block_color = rng.randint(2, 9)

    block_size = rng.randint(1, size - 1)
    dot_pos = rng.randint(0, size - 1)

    can_place_left = dot_pos >= block_size
    can_place_right = dot_pos + block_size < size

    if not (can_place_left or can_place_right):
        return None

    if can_place_left and can_place_right:
        side = rng.choice(("left", "right"))
    elif can_place_left:
        side = "left"
    else:
        side = "right"

    if side == "left":
        q_block_pos = rng.randint(0, dot_pos - block_size)
        a_block_pos = dot_pos - block_size
    else:
        q_block_pos = rng.randint(dot_pos + 1, size - block_size)
        a_block_pos = dot_pos + 1

    question = gen_field(size)
    question[dot_pos] = dot_color
    question = write_block(q_block_pos, [block_color] * block_size, question)

    answer = gen_field(size)
    answer[dot_pos] = dot_color
    answer = write_block(a_block_pos, [block_color] * block_size, answer)

    return {"input": question, "output": answer}


def transform_block_touch_dot(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    output = input_grid.copy()
    
    # Find dot position (color 1)
    dot_pos = None
    for i, color in enumerate(input_grid):
        if color == 1:
            dot_pos = i
            break
    
    if dot_pos is None:
        return input_grid  # No dot found, return original
    
    # Find block (any non-zero color except 1)
    block_start = None
    block_end = None
    block_color = None
    
    # Find start of block
    for i in range(size):
        if input_grid[i] != 0 and input_grid[i] != 1:
            block_start = i
            block_color = input_grid[i]
            break
    
    if block_start is None:
        return input_grid  # No block found, return original
    
    # Find end of block
    for i in range(block_start, size):
        if input_grid[i] != block_color:
            block_end = i
            break
    if block_end is None:
        block_end = size
    
    block_size = block_end - block_start
    
    # Clear the block from its current position
    for i in range(block_start, block_end):
        output[i] = 0
    
    # Check if block can be placed on either side
    can_place_left = dot_pos >= block_size
    can_place_right = dot_pos + block_size < size
    
    if not (can_place_left or can_place_right):
        return input_grid  # Can't place block on either side
    
    # Place block on the same side as it was originally
    if block_start < dot_pos:  # Block was originally to the left of dot
        new_start = dot_pos - block_size
    else:  # Block was originally to the right of dot
        new_start = dot_pos + 1
    
    # Place block at new position
    for i in range(block_size):
        output[new_start + i] = block_color
    
    return output


def task_block_touch_dot_n_pix(rng: Random, size: int, move_pix: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block moves move_pix pixels toward a dot."""
    dot_color = 2
    block_color = rng.randint(3, 9)

    block_size = rng.randint(1, size - 1)
    dot_pos = rng.randint(0, size - 1)

    can_place_left = dot_pos >= block_size
    can_place_right = dot_pos + block_size < size

    if not (can_place_left or can_place_right):
        return None

    if can_place_left and can_place_right:
        side = rng.choice(("left", "right"))
    elif can_place_left:
        side = "left"
    else:
        side = "right"

    if side == "left":
        q_block_pos = rng.randint(0, dot_pos - block_size)
        distance = (dot_pos - block_size) - q_block_pos
        move = min(distance, move_pix)
        a_block_pos = q_block_pos + move
    else:
        q_block_pos = rng.randint(dot_pos + 1, size - block_size)
        distance = q_block_pos - (dot_pos + 1)
        move = min(distance, move_pix)
        a_block_pos = q_block_pos - move

    question = gen_field(size)
    question[dot_pos] = dot_color
    question = write_block(q_block_pos, [block_color] * block_size, question)

    answer = gen_field(size)
    answer[dot_pos] = dot_color
    answer = write_block(a_block_pos, [block_color] * block_size, answer)

    return {"input": question, "output": answer}


def transform_block_touch_dot_n_pix(input_grid: list[int], move_pix: int) -> list[int]:
    size = len(input_grid)
    output = input_grid.copy()
    
    # Find dot position (color 2)
    dot_pos = None
    for i, color in enumerate(input_grid):
        if color == 2:
            dot_pos = i
            break
    
    if dot_pos is None:
        return input_grid  # No dot found, return original
    
    # Find block (any non-zero color except 2)
    block_start = None
    block_end = None
    block_color = None
    
    # Find start of block
    for i in range(size):
        if input_grid[i] != 0 and input_grid[i] != 2:
            block_start = i
            block_color = input_grid[i]
            break
    
    if block_start is None:
        return input_grid  # No block found, return original
    
    # Find end of block
    for i in range(block_start, size):
        if input_grid[i] != block_color:
            block_end = i
            break
    if block_end is None:
        block_end = size
    
    block_size = block_end - block_start
    
    # Clear the block from its current position
    for i in range(block_start, block_end):
        output[i] = 0
    
    # Calculate new position based on where the block is relative to the dot
    if block_start < dot_pos:  # Block is to the left of dot
        # Move right toward dot
        distance = (dot_pos - block_size) - block_start
        move = min(distance, move_pix)
        new_start = block_start + move
    else:  # Block is to the right of dot
        # Move left toward dot
        distance = block_start - (dot_pos + 1)
        move = min(distance, move_pix)
        new_start = block_start - move
    
    # Ensure the new position is valid
    if new_start < 0 or new_start + block_size > size:
        return input_grid  # Invalid position, return original
    
    # Place block at new position
    for i in range(block_size):
        output[new_start + i] = block_color
    
    return output


def task_block_scale_to_dot(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block scales to touch a dot (keeping one end fixed)."""
    dot_color = 2
    block_color = rng.randint(3, 9)

    block_size = rng.randint(1, size - 1)
    dot_pos = rng.randint(0, size - 1)

    can_place_left = dot_pos >= block_size
    can_place_right = dot_pos + block_size < size

    if not (can_place_left or can_place_right):
        return None

    if can_place_left and can_place_right:
        side = rng.choice(("left", "right"))
    elif can_place_left:
        side = "left"
    else:
        side = "right"

    if side == "left":
        q_block_pos = rng.randint(0, dot_pos - block_size)
        new_size = dot_pos - q_block_pos + 1
        a_block_pos = q_block_pos
    else:
        q_block_pos = rng.randint(dot_pos + 1, size - block_size)
        new_size = (q_block_pos + block_size) - dot_pos
        a_block_pos = dot_pos

    question = gen_field(size)
    question[dot_pos] = dot_color
    question = write_block(q_block_pos, [block_color] * block_size, question)

    answer = gen_field(size)
    answer[dot_pos] = dot_color
    answer = write_block(a_block_pos, [block_color] * new_size, answer)

    return {"input": question, "output": answer}


def transform_block_scale_to_dot(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    output = input_grid.copy()
    
    # Find dot position (color 2)
    dot_pos = None
    for i, color in enumerate(input_grid):
        if color == 2:
            dot_pos = i
            break
    
    if dot_pos is None:
        return input_grid  # No dot found, return original
    
    # Find block (any non-zero color except 2)
    block_start = None
    block_end = None
    block_color = None
    
    # Find start of block
    for i in range(size):
        if input_grid[i] != 0 and input_grid[i] != 2:
            block_start = i
            block_color = input_grid[i]
            break
    
    if block_start is None:
        return input_grid  # No block found, return original
    
    # Find end of block
    for i in range(block_start, size):
        if input_grid[i] != block_color:
            block_end = i
            break
    if block_end is None:
        block_end = size
    
    block_size = block_end - block_start
    
    # Clear the block from its current position
    for i in range(block_start, block_end):
        output[i] = 0
    
    # Calculate new size and position based on where the block is relative to the dot
    if block_start < dot_pos:  # Block is to the left of dot
        # Keep left end fixed, scale right to touch dot
        new_size = dot_pos - block_start + 1
        new_start = block_start
    else:  # Block is to the right of dot
        # Keep right end fixed, scale left to touch dot
        new_size = (block_end - 1) - dot_pos + 1
        new_start = dot_pos
    
    # Ensure the new position is valid
    if new_start < 0 or new_start + new_size > size:
        return input_grid  # Invalid position, return original
    
    # Place block at new position with new size
    for i in range(new_size):
        output[new_start + i] = block_color
    
    return output


def task_two_points_and_fill(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where space between two points of same color is filled with that color."""
    color = rng.randint(1, 9)

    pos1 = rng.randint(0, size - 1)
    pos2 = rng.randint(0, size - 1)
    if pos1 == pos2:
        return None

    pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

    question = gen_field(size)
    question[pos1] = color
    question[pos2] = color

    answer = question.copy()
    for i in range(pos1, pos2 + 1):
        answer[i] = color

    return {"input": question, "output": answer}


def transform_two_points_and_fill(input_grid: list[int], inverse: bool = False) -> list[int]:
    size = len(input_grid)
    # Find all nonzero positions and their colors
    nonzeros = [(i, v) for i, v in enumerate(input_grid) if v != 0]
    if not nonzeros:
        return input_grid.copy()

    # All nonzeros must share the same color
    _, color0 = nonzeros[0]
    if any(v != color0 for _, v in nonzeros):
        # mixed colors → nothing to do
        return input_grid.copy()

    if not inverse:
        # We expect exactly two points
        if len(nonzeros) != 2:
            return input_grid.copy()
        pos1, _ = nonzeros[0]
        pos2, _ = nonzeros[1]
        start, end = min(pos1, pos2), max(pos1, pos2)
        output = input_grid.copy()
        for i in range(start, end + 1):
            output[i] = color0
        return output

    else:
        # inverse=True: revert a filled segment back to just its two endpoints
        # we expect a contiguous run
        indices = [i for i, _ in nonzeros]
        start, end = min(indices), max(indices)
        # verify contiguity
        if end - start + 1 != len(indices):
            return input_grid.copy()
        # build grid with only endpoints
        output = [0] * size
        output[start] = color0
        output[end] = color0
        return output



def task_reflect_block_with_border_pixel(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block with a border pixel is reflected."""
    block_size = rng.randint(2, size)

    c1 = rng.randint(1, 9)
    c2 = rng.choice(tuple(c for c in range(1, 9) if c != c1))

    side = "left" if rng.random() < 0.5 else "right"
    pos = rng.randint(0, size - block_size)

    block = [c1] * block_size
    if side == "left":
        block[0] = c2
    else:
        block[block_size - 1] = c2

    question = write_block(pos, block, gen_field(size))
    reversed_block = block[::-1]  # Reverse the block
    answer = write_block(pos, reversed_block, gen_field(size))

    return {"input": question, "output": answer}


def transform_reflect_block_with_border_pixel(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    output = input_grid.copy()
    
    # Find the block by looking for a sequence with a different color on either end
    block_start = None
    block_end = None
    block_colors = []
    
    # Scan through the grid to find potential blocks
    i = 0
    while i < size:
        if input_grid[i] != 0:  # Found a non-zero pixel
            # Check if this could be the start of a block
            if i + 1 < size and input_grid[i + 1] != 0:
                # Found potential block start
                block_start = i
                block_colors = [input_grid[i]]
                
                # Find the end of the block
                j = i + 1
                while j < size and input_grid[j] != 0:
                    block_colors.append(input_grid[j])
                    j += 1
                block_end = j - 1
                
                # Check if this is a valid block with a border pixel
                if len(block_colors) >= 2 and (block_colors[0] != block_colors[1] or block_colors[-1] != block_colors[-2]):
                    # Found a valid block with border pixel
                    break
                else:
                    # Not a valid block, continue searching
                    block_start = None
                    block_end = None
                    block_colors = []
                    i = j
            else:
                i += 1
        else:
            i += 1
    
    # If we found a valid block, reflect it
    if block_start is not None and block_end is not None:
        # Reverse the block colors
        reversed_colors = block_colors[::-1]
        
        # Write the reversed block back to the same position
        for i, color in enumerate(reversed_colors):
            output[block_start + i] = color
    
    return output


def task_reflect_block_with_border_pixel_random(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a random-colored block with a border pixel is reflected."""
    block_size = rng.randint(2, size)

    side = "left" if rng.random() < 0.5 else "right"
    pos = rng.randint(0, size - block_size)

    border_color = rng.randint(1, 9)
    other_colors = tuple(c for c in range(1, 9) if c != border_color)
    block = [rng.choice(other_colors) for _ in range(block_size)]

    if side == "left":
        block[0] = border_color
    else:
        block[block_size - 1] = border_color

    question = write_block(pos, block, gen_field(size))
    reversed_block = block[::-1]  # Reverse the block
    answer = write_block(pos, reversed_block, gen_field(size))

    return {"input": question, "output": answer}


def transform_reflect_block_with_border_pixel_random(input_grid: list[int]) -> list[int]:
    size = len(input_grid)

    # 1. Locate the block start (first nonzero)
    block_start = None
    for i, v in enumerate(input_grid):
        if v != 0:
            block_start = i
            break
    if block_start is None:
        # no block found
        return input_grid.copy()

    # 2. Measure the block (collect until zero or end)
    block = []
    idx = block_start
    while idx < size and input_grid[idx] != 0:
        block.append(input_grid[idx])
        idx += 1
    block_size = len(block)

    # 3. Reverse the block
    reversed_block = block[::-1]

    # 4. Build output: zeros everywhere, then write reversed block
    output = [0] * size
    for offset, color in enumerate(reversed_block):
        output[block_start + offset] = color

    return output


def task_reflect_block_around_dot(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block is reflected around a dot."""
    dot_color = 2

    dot_pos = rng.randint(0, size - 1)
    block_size = rng.randint(1, size - 1)
    block_pos = rng.randint(0, size - block_size)
    block_end = block_pos + block_size - 1

    # Check if block is strictly to left or right of dot
    strictly_left = block_end < dot_pos
    strictly_right = block_pos > dot_pos

    if not (strictly_left or strictly_right):
        return None

    block_color = rng.randint(3, 9)  # Different from dot color
    block = [block_color] * block_size

    # Calculate reflection bounds
    min_reflect = 2 * dot_pos - block_end
    max_reflect = 2 * dot_pos - block_pos
    if min_reflect < 0 or max_reflect >= size:
        return None

    question = gen_field(size)
    question = write_block(block_pos, block, question)
    question[dot_pos] = dot_color

    answer = gen_field(size)
    answer[dot_pos] = dot_color
    for i in range(block_size):
        reflect_idx = 2 * dot_pos - (block_pos + i)
        answer[reflect_idx] = block[i]

    return {"input": question, "output": answer}


def transform_reflect_block_around_dot(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    dot_color = 2

    # Find the dot position
    try:
        dot_pos = input_grid.index(dot_color)
    except ValueError:
        raise ValueError("Input grid must contain exactly one dot (value 2).")

    # Identify the block: all positions with a color >= 3
    block_indices = [i for i, v in enumerate(input_grid) if v >= 3]
    if not block_indices:
        # No block => nothing to reflect
        return input_grid.copy()

    block_start = min(block_indices)
    block_end   = max(block_indices)
    block_size  = block_end - block_start + 1

    # Extract block colors in order
    block = [input_grid[i] for i in range(block_start, block_end + 1)]

    # Build the answer as a fresh zero-field
    answer = [0] * size
    # Copy the dot
    answer[dot_pos] = dot_color

    # Reflect each element of the block
    # reflection: new_idx = 2*dot_pos - original_idx
    for offset, color in enumerate(block):
        orig_idx   = block_start + offset
        reflect_idx = 2 * dot_pos - orig_idx
        if not (0 <= reflect_idx < size):
            raise ValueError("Reflection would go out of bounds.")
        answer[reflect_idx] = color

    return answer


def task_block_and_noise_remove(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where noise around a block needs to be removed."""
    block_size = rng.randint(2, size)

    block_pos = rng.randint(0, size - block_size)
    color = rng.randint(1, 9)

    # Create field with block
    field = gen_field(size)
    for i in range(block_size):
        field[block_pos + i] = color

    # Track forbidden positions for noise
    forbidden = [False] * size
    for i in range(block_pos, block_pos + block_size):
        forbidden[i] = True
    if block_pos > 0:
        forbidden[block_pos - 1] = True
    if block_pos + block_size < size:
        forbidden[block_pos + block_size] = True

    # Add noise
    noise_count = rng.randint(1, 3)
    noise_positions = []

    for _ in range(noise_count):
        allowed = tuple(i for i in range(size) if not forbidden[i])
        if not allowed:
            break
        noise_pos = rng.choice(allowed)
        noise_positions.append(noise_pos)
        field[noise_pos] = color
        forbidden[noise_pos] = True
        if noise_pos > 0:
            forbidden[noise_pos - 1] = True
        if noise_pos + 1 < size:
            forbidden[noise_pos + 1] = True

    if len(noise_positions) < noise_count:
        return None

    question = field
    answer = field.copy()
    for pos in noise_positions:
        answer[pos] = 0

    return {"input": question, "output": answer}


def transform_block_and_noise_remove(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    # Identify the block color (the only non-zero value in the grid)
    nonzero_colors = {v for v in input_grid if v != 0}
    if not nonzero_colors:
        # nothing to remove
        return input_grid.copy()
    if len(nonzero_colors) > 1:
        raise ValueError("Expected only one non-zero color in the input grid.")
    color = nonzero_colors.pop()

    # Find all indices of that color
    indices = [i for i, v in enumerate(input_grid) if v == color]
    if not indices:
        return input_grid.copy()

    # Group into contiguous runs: list of (start_idx, length)
    runs = []
    run_start = indices[0]
    run_len = 1
    for prev, curr in zip(indices, indices[1:]):
        if curr == prev + 1:
            run_len += 1
        else:
            runs.append((run_start, run_len))
            run_start = curr
            run_len = 1
    runs.append((run_start, run_len))

    # The true block is the longest run (length ≥ 2)
    block_run = max(runs, key=lambda x: x[1])
    if block_run[1] < 2:
        raise ValueError("No contiguous block of size ≥ 2 found.")
    # Noise runs are all other runs of length == 1
    noise_positions = [start for start, length in runs if length == 1]

    # Build the answer: copy input, zero out noise positions
    output = input_grid.copy()
    for pos in noise_positions:
        output[pos] = 0

    return output



def task_block_and_noise_remove_inside(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where noise inside a block needs to be removed."""
    if size <= 6:
        return None

    block_size = rng.randint(6, size)

    block_pos = rng.randint(0, size - block_size)
    color = rng.randint(1, 9)

    # Create field with block
    field = gen_field(size)
    for i in range(block_size):
        field[block_pos + i] = color

    # Add noise inside block
    max_noise = max(1, (block_size // 2) - 1)
    noise_count = rng.randint(1, max_noise)

    positions = list(range(block_size))
    rng.shuffle(positions)
    noise_positions = positions[:noise_count]

    for offset in noise_positions:
        pos = block_pos + offset
        noise_color = rng.randint(1, 9)
        while noise_color == color:
            noise_color = rng.randint(1, 9)
        field[pos] = noise_color

    question = field
    answer = field.copy()
    for offset in noise_positions:
        answer[block_pos + offset] = color

    return {"input": question, "output": answer}

def transform_block_and_noise_remove_inside(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    # Find the block region bounds (first/last non-zero)
    nonzeros = [i for i, v in enumerate(input_grid) if v != 0]
    if not nonzeros:
        return input_grid.copy()

    start, end = nonzeros[0], nonzeros[-1]
    region = input_grid[start : end + 1]

    # The true block color C is the majority value in region
    # (block_size > 6 ensures that C appears more than any noise)
    counts = {}
    for v in region:
        counts[v] = counts.get(v, 0) + 1
    # Pick the value with max count
    block_color = max(counts, key=counts.get)

    # Build output: zeroes outside, and region all set to block_color
    output = input_grid.copy()
    for i in range(start, end + 1):
        output[i] = block_color

    return output


def task_copy_block_to_dots(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block pattern is copied to dot positions."""
    block_size = 3 if rng.random() < 0.5 else 5
    if block_size >= size:
        return None

    color = rng.randint(1, 9)
    block = [color] * block_size

    # Generate dots with minimum distance to prevent overlap
    min_gap = block_size
    dot_positions = []
    pos = block_size + block_size // 2 + 1

    while pos <= size - block_size:
        if rng.random() < 0.5:  # Control dot density
            dot_positions.append(pos)
            pos += min_gap
        pos += 1

    if not dot_positions:
        return None

    question = gen_field(size)
    question = write_block(0, block, question)
    for pos in dot_positions:
        question[pos] = color

    answer = gen_field(size)
    answer = write_block(0, block, answer)
    for pos in dot_positions:
        block_start = pos - block_size // 2
        answer = write_block(block_start, block, answer)

    return {"input": question, "output": answer}


def transform_copy_block_to_dots(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    # 1. Determine the block color and size at the left
    block_color = input_grid[0]
    if block_color == 0:
        raise ValueError("No block found at index 0")
    block_size = 1
    while block_size < size and input_grid[block_size] == block_color:
        block_size += 1

    # 2. Locate the dot positions (same color, but outside the initial block)
    dot_positions = [
        i for i, v in enumerate(input_grid)
        if v == block_color and i >= block_size
    ]

    # 3. Build the output grid
    answer = [0] * size

    # 3a. Copy the original block at 0
    for i in range(block_size):
        answer[i] = block_color

    # 3b. For each dot, copy the block centered at that dot
    half = block_size // 2
    for pos in dot_positions:
        start = pos - half
        # write block_size cells starting at 'start'
        for j in range(block_size):
            idx = start + j
            if 0 <= idx < size:
                answer[idx] = block_color

    return answer

def task_copy_block_to_dots_colors(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block pattern is copied to dot positions with matching colors."""
    block_size = 3 if rng.random() < 0.5 else 5
    if block_size >= size:
        return None

    block_color = rng.randint(1, 9)
    block = [block_color] * block_size

    # Generate dots with minimum distance to prevent overlap
    min_gap = block_size
    dot_positions = []
    dot_colors = []
    pos = block_size + block_size // 2 + 1

    while pos <= size - block_size:
        if rng.random() < 0.5:
            dot_color = rng.randint(1, 9)
            dot_positions.append(pos)
            dot_colors.append(dot_color)
            pos += min_gap
        pos += 1

    if not dot_positions:
        return None

    question = gen_field(size)
    question = write_block(0, block, question)
    for i, pos in enumerate(dot_positions):
        question[pos] = dot_colors[i]

    answer = gen_field(size)
    answer = write_block(0, block, answer)
    for i, pos in enumerate(dot_positions):
        block_start = pos - block_size // 2
        colored_block = [dot_colors[i]] * block_size
        answer = write_block(block_start, colored_block, answer)

    return {"input": question, "output": answer}


def transform_copy_block_to_dots_colors(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    # 1. Determine the initial block size and color at the left
    block_color = input_grid[0]
    block_size = 1
    while block_size < size and input_grid[block_size] == block_color:
        block_size += 1

    # 2. Find dot positions and their colors (non-zero outside the initial block)
    dot_positions = []
    dot_colors = []
    for i in range(block_size, size):
        v = input_grid[i]
        if v != 0:
            dot_positions.append(i)
            dot_colors.append(v)

    # 3. Build the output grid
    output = [0] * size

    # 3a. Copy the original block at index 0
    for i in range(block_size):
        output[i] = block_color

    # 3b. For each dot, copy a block of its color centered at that dot
    half = block_size // 2
    for pos, color in zip(dot_positions, dot_colors):
        start = pos - half
        for j in range(block_size):
            idx = start + j
            if 0 <= idx < size:
                output[idx] = color

    return output


def task_paint_biggest_block(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where the largest block is painted a different color."""
    target_color = 1
    initial_color = rng.randint(2, 9)

    # Generate random blocks
    question = gen_field(size)
    blocks = []
    pos = 0

    while pos < size:
        if rng.random() < 0.4 and size - pos >= 2:
            block_size = rng.randint(2, min(size - pos, 6))
            blocks.append((pos, block_size))
            for i in range(block_size):
                question[pos + i] = initial_color
            pos += block_size + 1
        else:
            pos += 1

    if len(blocks) < 2:
        return None

    # Find biggest block
    biggest_pos, biggest_size = max(blocks, key=lambda x: x[1])

    # Check if there are multiple blocks of the same size
    biggest_count = sum(1 for _, size in blocks if size == biggest_size)
    if biggest_count > 1:
        return None

    answer = question.copy()
    for i in range(biggest_size):
        answer[biggest_pos + i] = target_color

    return {"input": question, "output": answer}


def transform_paint_biggest_block(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    # 1. Identify all contiguous non-zero runs (blocks)
    runs = []
    i = 0
    while i < size:
        if input_grid[i] != 0:
            start = i
            while i < size and input_grid[i] != 0:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1

    # If fewer than two blocks, nothing to do
    if len(runs) < 2:
        return input_grid.copy()

    # 2. Find the unique largest block
    biggest_start, biggest_len = max(runs, key=lambda x: x[1])

    # 3. Paint that block to color 1
    output = input_grid.copy()
    for offset in range(biggest_len):
        output[biggest_start + offset] = 1

    return output


def task_sort_blocks_by_size(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where blocks are sorted by size with 1 pixel gaps."""
    color = rng.randint(1, 9)
    blocks = []
    pos = 0

    # Generate random blocks with random sizes
    while pos < size:
        if rng.random() < 0.4 and size - pos >= 2:
            block_size = rng.randint(1, min(size - pos, 6))
            blocks.append((pos, block_size))
            pos += block_size + rng.randint(1, 4)  # Random gaps
        else:
            pos += 1

    if len(blocks) < 2:
        return None

    # Create input field
    question = gen_field(size)
    for pos, block_size in blocks:
        for i in range(block_size):
            question[pos + i] = color

    # Sort blocks by size
    blocks.sort(key=lambda x: x[1])

    # Check if sorted blocks fit with gaps
    total_space = sum(size for _, size in blocks) + len(blocks) - 1
    if total_space > size:
        return None

    # Create answer field with sorted blocks
    answer = gen_field(size)
    current_pos = 0

    for _, block_size in blocks:
        for i in range(block_size):
            answer[current_pos + i] = color
        current_pos += block_size + 1  # One pixel gap

    return {"input": question, "output": answer}


def transform_sort_blocks_by_size(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    # 1. Identify all contiguous non-zero runs (blocks) and their lengths
    runs = []
    i = 0
    while i < size:
        if input_grid[i] != 0:
            start = i
            while i < size and input_grid[i] != 0:
                i += 1
            runs.append(i - start)
        else:
            i += 1

    # If fewer than two blocks, nothing to sort
    if len(runs) < 2:
        return input_grid.copy()

    # 2. Sort the block lengths
    sorted_lengths = sorted(runs)

    # 3. Rebuild the output grid
    output = [0] * size
    current_pos = 0
    # We assume all blocks use the same color; grab it from the first block in input
    block_color = next(v for v in input_grid if v != 0)

    for length in sorted_lengths:
        # Place a block of this length
        for offset in range(length):
            output[current_pos + offset] = block_color
        # Advance past block plus one gap
        current_pos += length + 1
        # If we run out of space, we just stop (shouldn't happen if input was valid)
        if current_pos >= size:
            break

    return output


def task_sort_complete_sequence(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a complete sequence of block sizes is sorted."""
    # Calculate max possible block size given total array size
    max_size = 1
    total_space = 0
    while total_space + max_size + 1 <= size:
        total_space += max_size + 1
        max_size += 1
    max_size -= 1

    if max_size < 2:
        return None

    color = rng.randint(1, 9)

    # Create sequence of all sizes from 1 to max_size
    blocks = list(range(1, max_size + 1))
    rng.shuffle(blocks)

    # Create input field with shuffled blocks
    question = gen_field(size)
    pos = 0
    for block_size in blocks:
        for i in range(block_size):
            question[pos + i] = color
        pos += block_size + 1

    # Create answer field with sorted blocks
    answer = gen_field(size)
    pos = 0
    for block_size in range(1, max_size + 1):
        for i in range(block_size):
            answer[pos + i] = color
        pos += block_size + 1

    return {"input": question, "output": answer}


def transform_sort_complete_sequence(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    # 1. Detect the block color (first non-zero)
    block_color = next((v for v in input_grid if v != 0), None)
    if block_color is None:
        return input_grid.copy()

    # 2. Extract all block sizes in left-to-right order
    runs = []
    i = 0
    while i < size:
        if input_grid[i] == block_color:
            start = i
            while i < size and input_grid[i] == block_color:
                i += 1
            runs.append(i - start)
        else:
            i += 1

    # If fewer than two blocks, nothing to do
    if len(runs) < 2:
        return input_grid.copy()

    # 3. Sort the sizes ascending
    runs_sorted = sorted(runs)

    # 4. Build the output grid
    output = [0] * size
    pos = 0
    for block_size in runs_sorted:
        # place block
        for offset in range(block_size):
            output[pos + offset] = block_color
        # advance past block + one-gap
        pos += block_size + 1
        if pos >= size:
            break

    return output


def task_recolor_blocks_by_size(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where two blocks are recolored based on their size."""
    # Generate two different random sizes
    size1 = rng.randint(2, 8)
    size2 = rng.randint(2, 8)
    while size2 == size1:
        size2 = rng.randint(2, 8)

    # Ensure both blocks fit with at least 1 gap
    if size1 + size2 + 1 > size:
        return None

    # Place blocks with gap
    pos1 = rng.randint(0, size - (size1 + size2 + 1))
    pos2 = rng.randint(pos1 + size1 + 1, size - size2)

    # Create input field with both blocks color 3
    question = gen_field(size)
    for i in range(size1):
        question[pos1 + i] = 3
    for i in range(size2):
        question[pos2 + i] = 3

    # Create answer field with recolored blocks
    answer = question.copy()
    if size1 > size2:
        for i in range(size1):
            answer[pos1 + i] = 1
        for i in range(size2):
            answer[pos2 + i] = 2
    else:
        for i in range(size1):
            answer[pos1 + i] = 2
        for i in range(size2):
            answer[pos2 + i] = 1

    return {"input": question, "output": answer}


def transform_recolor_blocks_by_size(input_grid: list[int]) -> list[int]:
    size = len(input_grid)

    # 1. Identify all contiguous runs of color 3
    runs = []
    i = 0
    while i < size:
        if input_grid[i] == 3:
            start = i
            while i < size and input_grid[i] == 3:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1

    # Expect exactly two runs
    if len(runs) != 2:
        raise ValueError("Input must contain exactly two blocks of color 3.")

    # 2. Determine which run is larger
    (s1, len1), (s2, len2) = runs
    # Assign target colors based on size
    if len1 > len2:
        big_start, big_len, big_color = s1, len1, 1
        small_start, small_len, small_color = s2, len2, 2
    else:
        big_start, big_len, big_color = s2, len2, 1
        small_start, small_len, small_color = s1, len1, 2

    # 3. Build the output by copying and recoloring
    output = input_grid.copy()
    # Recolor big block
    for offset in range(big_len):
        output[big_start + offset] = big_color
    # Recolor small block
    for offset in range(small_len):
        output[small_start + offset] = small_color

    return output



def task_gravity_one_step(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where non-zero elements move one step left if possible."""
    question = [rng.randint(1, 9) if rng.random() < 0.5 else 0 for _ in range(size)]
    answer = question.copy()

    # Move each non-zero pixel one step left if possible
    for i in range(1, size):
        if answer[i] != 0 and answer[i - 1] == 0:
            answer[i - 1] = answer[i]
            answer[i] = 0

    return {"input": question, "output": answer}

def transform_gravity_one_step(input_grid: list[int], direction: str = "right") -> list[int]:
    size = len(input_grid)
    output = input_grid.copy()
    
    if direction == "right":
        # scan leftwards: from index 1 to end
        for i in range(1, size):
            if output[i] != 0 and output[i - 1] == 0:
                output[i - 1] = output[i]
                output[i] = 0

    elif direction == "left":
        # scan rightwards: from index size-2 down to 0
        for i in range(size - 2, -1, -1):
            if output[i] != 0 and output[i + 1] == 0:
                output[i + 1] = output[i]
                output[i] = 0

    else:
        raise ValueError("direction must be 'left' or 'right'")

    return output



def task_move_block_by_own_size(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block moves right by its own size."""
    block_size = rng.randint(1, size // 2)  # Ensure space for movement
    pos = rng.randint(0, size - block_size * 2)  # Space for block and movement
    color = rng.randint(1, 9)

    question = gen_field(size)
    block = [color] * block_size
    question = write_block(pos, block, question)

    answer = write_block(pos + block_size, block, gen_field(size))

    return {"input": question, "output": answer}


def transform_move_block_by_own_size(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    # 1. Find the block: first non-zero index and its length
    start = None
    for i, v in enumerate(input_grid):
        if v != 0:
            start = i
            break
    if start is None:
        # no block, return as is
        return input_grid.copy()
    # find end of block
    end = start
    while end < size and input_grid[end] != 0:
        end += 1
    block_size = end - start
    block_color = input_grid[start]

    # 2. Build output: all zeros, then place block at start + block_size
    output = [0] * size
    new_start = start + block_size
    for offset in range(block_size):
        idx = new_start + offset
        if idx < size:
            output[idx] = block_color

    return output



def task_change_to_five(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where all non-zero colors change to 5."""
    density = 0.5
    question = [rng.randint(1, 9) if rng.random() < density else 0 for _ in range(size)]
    answer = [5 if x != 0 else 0 for x in question]

    return {"input": question, "output": answer}

def transform_change_to_five(input_grid: list[int]) -> list[int]:
    return [5 if x != 0 else 0 for x in input_grid]


def task_recolor_blocks_from_palette(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where blocks are recolored using a color palette."""
    # Generate blocks of same size
    block_size = rng.randint(2, 4)
    blocks = []
    pos = 0

    while pos + block_size <= size:
        if rng.random() < 0.4:
            blocks.append(pos)
            pos += block_size + 1
        else:
            pos += 1

    # Ensure we have space for palette
    while blocks and blocks[-1] + block_size + len(blocks) + 1 >= size:
        blocks.pop()

    if not blocks:
        return None

    # Shift blocks right to make room for palette
    palette_size = len(blocks)
    blocks = [pos + palette_size + 1 for pos in blocks]

    # Generate color palette
    colors = []
    for _ in range(len(blocks)):
        while True:
            color = rng.randint(1, 9)
            if color not in colors:
                colors.append(color)
                break

    # Create question with color palette and blocks
    question = gen_field(size)

    # Place color palette at start
    for i, color in enumerate(colors):
        question[i] = color

    # Place blocks of color 5
    for block_pos in blocks:
        for i in range(block_size):
            question[block_pos + i] = 5

    # Create answer with recolored blocks
    answer = question.copy()
    for block_idx, block_pos in enumerate(blocks):
        color = colors[block_idx]
        for i in range(block_size):
            answer[block_pos + i] = color

    return {"input": question, "output": answer}


def transform_recolor_blocks_from_palette(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    # 1. Read off the palette: all nonzero values from 0 until the first zero.
    palette = []
    for v in input_grid:
        if v == 0:
            break
        palette.append(v)
    P = len(palette)
    if P == 0:
        return input_grid.copy()

    # 2. Determine block size B by finding the first run of 5s after index P
    #    (there must be at least one block of 5s).
    i = P + 1
    # skip any extra zeros
    while i < size and input_grid[i] != 5:
        i += 1
    if i >= size:
        return input_grid.copy()  # no blocks found
    # now count block length B
    B = 0
    while i + B < size and input_grid[i + B] == 5:
        B += 1

    # 3. Collect the start‐indices of all runs of exactly B fives
    block_starts = []
    idx = P + 1
    while idx < size:
        if input_grid[idx] == 5:
            # check if this run is length B
            # (we assume well‐formed input, so every run of 5s has length B)
            block_starts.append(idx)
            idx += B
        else:
            idx += 1

    # 4. Recolor each block i with palette[i]
    output = input_grid.copy()
    for i, start in enumerate(block_starts):
        color = palette[i]
        for offset in range(B):
            output[start + offset] = color

    return output



def task_duplicate_block_from_seeds(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a block is duplicated from seed pixels."""
    block_size = rng.randint(2, 4)
    if block_size + 1 >= size:
        return None
    if size <= 3 + block_size:
        return None

    # Position block with space for seeds
    block_pos = rng.randint(2, size - block_size - 2)

    # Decide seed placement
    left_seed = False
    right_seed = False
    while not left_seed and not right_seed:
        left_seed = rng.random() < 0.5
        right_seed = rng.random() < 0.5

    # Create input
    question = gen_field(size)

    # Place main block
    for i in range(block_size):
        question[block_pos + i] = 1

    # Place seeds with gaps
    seeds = []
    if left_seed:
        color = rng.randint(1, 9)
        question[block_pos - 2] = color
        seeds.append(("left", block_pos - 2, color))
    if right_seed:
        color = rng.randint(1, 9)
        question[block_pos + block_size + 1] = color
        seeds.append(("right", block_pos + block_size + 1, color))

    # Create answer with duplicated blocks
    answer = question.copy()

    for side, seed_pos, color in seeds:
        if side == "left":
            # For left seed, blocks end at seed
            end_pos = seed_pos
            while end_pos >= 0:
                start_pos = end_pos - block_size + 1
                for pos in range(max(0, start_pos), end_pos + 1):
                    answer[pos] = color
                if start_pos < 1:
                    break
                end_pos = start_pos - 2  # -1 for gap
        else:  # side == "right"
            # For right seed, blocks start at seed
            start_pos = seed_pos
            while start_pos < size:
                for offset in range(min(block_size, size - start_pos)):
                    answer[start_pos + offset] = color
                if start_pos + block_size + 1 >= size:
                    break
                start_pos = start_pos + block_size + 1  # +1 for gap

    return {"input": question, "output": answer}


def transform_duplicate_block_from_seeds(input_grid: list[int]) -> list[int]:
    size = len(input_grid)

    # 1. Locate the main block: the first run of 1's of length ≥ 2
    block_start = None
    B = 0
    i = 0
    while i < size:
        if input_grid[i] == 1:
            # count contiguous 1's
            j = i
            while j < size and input_grid[j] == 1:
                j += 1
            run_len = j - i
            if run_len >= 2:
                block_start, B = i, run_len
                break
            else:
                i = j
        else:
            i += 1

    # no valid block found
    if block_start is None:
        return input_grid.copy()

    block_end = block_start + B - 1

    # 2. Check for seeds at exactly two positions
    seeds = []
    left_pos = block_start - 2
    if 0 <= left_pos < size and input_grid[left_pos] != 0:
        seeds.append(("left", left_pos, input_grid[left_pos]))
    right_pos = block_end + 2
    if 0 <= right_pos < size and input_grid[right_pos] != 0:
        seeds.append(("right", right_pos, input_grid[right_pos]))

    # 3. Start from the original grid and apply duplications
    output = input_grid.copy()

    for side, seed_pos, color in seeds:
        if side == "left":
            end_pos = seed_pos
            while end_pos >= 0:
                start_pos = end_pos - B + 1
                # paint from max(0,start_pos) to end_pos
                for p in range(max(0, start_pos), end_pos + 1):
                    output[p] = color
                if start_pos < 1:
                    break
                # move to next block location (gap of 1)
                end_pos = start_pos - 2

        else:  # "right"
            start_pos = seed_pos
            while start_pos < size:
                # paint B cells from start_pos
                for offset in range(min(B, size - start_pos)):
                    output[start_pos + offset] = color
                if start_pos + B + 1 >= size:
                    break
                # move to next block location (gap of 1)
                start_pos += B + 1

    return output



def task_fill_from_pixel(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a pixel fills in one direction until hitting another pixel."""
    if size < 8:
        return None

    block_size = rng.randint(3, size - 5)

    # Position block with space for seed
    block_pos = rng.randint(2, size - block_size - 2)

    # Create input
    question = gen_field(size)

    # Place main block
    block_color = rng.randint(1, 9)
    for i in range(block_size):
        question[block_pos + i] = block_color

    # Place seed pixel and determine fill direction
    seed_color = rng.randint(1, 8)
    if seed_color >= block_color:
        seed_color += 1

    is_left = rng.random() < 0.5

    if is_left:
        question[block_pos - 1] = seed_color
    else:
        question[block_pos + block_size] = seed_color

    # Create answer with fill
    answer = question.copy()

    if is_left:
        # Fill from seed to left border
        for i in range(block_pos):
            answer[i] = seed_color
    else:
        # Fill from seed to right border
        for i in range(block_pos + block_size, size):
            answer[i] = seed_color

    return {"input": question, "output": answer}


def transform_fill_from_pixel(input_grid: list[int]) -> list[int]:
    size = len(input_grid)

    # 1. Identify the main block: the contiguous run of length ≥ 3
    runs = []
    i = 0
    while i < size:
        if input_grid[i] != 0:
            start = i
            while i < size and input_grid[i] == input_grid[start]:
                i += 1
            length = i - start
            runs.append((start, length, input_grid[start]))
        else:
            i += 1

    # Find the run with length >= 3 (the main block)
    block_runs = [(s, l, c) for (s, l, c) in runs if l >= 3]
    if not block_runs:
        # nothing to fill
        return input_grid.copy()
    block_start, block_len, block_color = block_runs[0]
    block_end = block_start + block_len - 1

    # 2. Find the seed adjacent to the block
    seed_pos = None
    seed_color = None
    # Check left
    if block_start - 1 >= 0 and input_grid[block_start - 1] not in (0, block_color):
        seed_pos = block_start - 1
        seed_color = input_grid[seed_pos]
        direction = "left"
    # Check right
    elif block_end + 1 < size and input_grid[block_end + 1] not in (0, block_color):
        seed_pos = block_end + 1
        seed_color = input_grid[seed_pos]
        direction = "right"
    else:
        # no valid seed found
        return input_grid.copy()

    # 3. Build the output by filling from seed to the border
    output = input_grid.copy()
    if direction == "left":
        for idx in range(0, block_start):
            output[idx] = seed_color
    else:  # right
        for idx in range(block_end + 1, size):
            output[idx] = seed_color

    return output


def task_mark_size_two_blocks(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where size-2 blocks are marked with surrounding pixels."""
    if size < 8:
        return None

    # Start with one size-2 block
    blocks = [2]
    pos = 4  # Space for first block (2) + gap (2)

    # Generate more blocks
    while pos < size:
        if rng.random() < 0.4:
            block_size = rng.randint(1, 3)
            if pos + block_size <= size:
                blocks.append(block_size)
            pos += block_size + 2  # block + gap
        else:
            blocks.append(0)
            pos += 1

    # Shuffle block sizes
    rng.shuffle(blocks)

    # Assign positions with proper gaps
    block_positions = []
    pos = 0

    for block_size in blocks:
        if block_size == 0:
            pos += 1
        else:
            block_positions.append((pos, block_size))
            pos += block_size + 2  # Move past block + gap

    # Create input with blocks
    question = gen_field(size)
    for pos, block_size in block_positions:
        block_color = rng.randint(1, 8)
        if block_color >= 3:  # avoid marker color 3
            block_color += 1
        for i in range(block_size):
            question[pos + i] = block_color

    # Create answer with markers
    answer = question.copy()
    for pos, block_size in block_positions:
        if block_size == 2:
            if pos > 0:
                answer[pos - 1] = 3
            if pos + block_size < size:
                answer[pos + block_size] = 3

    return {"input": question, "output": answer}


def transform_mark_size_two_blocks(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    output = input_grid.copy()

    i = 0
    while i < size:
        # detect start of a non-zero run
        if input_grid[i] != 0:
            start = i
            # advance to end of run
            while i < size and input_grid[i] != 0:
                i += 1
            length = i - start
            # if it's size‑2, place markers
            if length == 2:
                if start - 1 >= 0:
                    output[start - 1] = 3
                end = start + length - 1
                if end + 1 < size:
                    output[end + 1] = 3
        else:
            i += 1

    return output


def task_fill_until_collision(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where pixels fill empty space until collision."""
    # At least 4 positions for meaningful puzzle
    if size < 4:
        return None

    is_left = rng.random() < 0.5
    question = gen_field(size)

    # Place the side marker
    if is_left:
        question[0] = 5
    else:
        question[size - 1] = 5

    # Place 2-4 random pixels
    num_pixels = rng.randint(2, 4)
    positions = []

    if is_left:
        # Skip first position
        for _ in range(num_pixels):
            while True:
                pos = rng.randint(1, size - 1)
                if pos not in positions:
                    positions.append(pos)
                    break
    else:
        # Skip last position
        for _ in range(num_pixels):
            while True:
                pos = rng.randint(0, size - 2)
                if pos not in positions:
                    positions.append(pos)
                    break

    # Color random pixels
    for pos in positions:
        c = rng.randint(1, 8)
        if c >= 5:  # don't use side marker color 5
            c += 1
        question[pos] = c

    positions.sort()

    # Create answer
    answer = question.copy()

    if is_left:
        # Fill right from each pixel
        prev_pos = 0  # Start from marker
        for pos in positions:
            color = question[pos]
            # Fill from previous position to current
            for i in range(prev_pos + 1, pos):
                answer[i] = color
            prev_pos = pos
    else:
        # Fill left from each pixel
        prev_pos = size - 1  # Start from marker
        for pos in reversed(positions):
            color = question[pos]
            # Fill from current position to previous
            for i in range(pos + 1, prev_pos):
                answer[i] = color
            prev_pos = pos

    return {"input": question, "output": answer}


def transform_fill_until_collision(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    # 1. Determine direction and marker position
    if input_grid[0] == 5:
        is_left = True
        marker_pos = 0
    elif input_grid[-1] == 5:
        is_left = False
        marker_pos = size - 1
    else:
        # no valid marker, return copy
        return input_grid.copy()

    # 2. Identify all pixel positions (non-zero except the marker)
    pixel_positions = [i for i, v in enumerate(input_grid) if v != 0 and i != marker_pos]
    pixel_positions.sort()

    # 3. Build the output by filling between marker/pixels
    output = input_grid.copy()
    if is_left:
        prev = marker_pos
        for pos in pixel_positions:
            color = input_grid[pos]
            for i in range(prev + 1, pos):
                output[i] = color
            prev = pos
    else:
        prev = marker_pos
        for pos in reversed(pixel_positions):
            color = input_grid[pos]
            for i in range(pos + 1, prev):
                output[i] = color
            prev = pos

    return output



def task_repeat_pattern_full(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where a pattern is repeated to fill the space."""
    # Generate initial pattern
    pattern_size = rng.randint(2, 5)
    pattern = [rng.randint(1, 9) for _ in range(pattern_size)]

    # Calculate total size needed for 2 repetitions
    double_size = pattern_size * 2
    if double_size >= size:
        return None

    # Create input with 2 repetitions
    question = gen_field(size)
    for i in range(pattern_size):
        question[i] = pattern[i]
        question[i + pattern_size] = pattern[i]

    # Create answer with maximum repetitions
    answer = gen_field(size)
    pos = 0
    while pos + pattern_size <= size:
        for i in range(pattern_size):
            answer[pos + i] = pattern[i]
        pos += pattern_size

    # Fill remaining space (if any) with pattern elements
    for i in range(pos, size):
        answer[i] = pattern[i - pos]

    return {"input": question, "output": answer}


def transform_repeat_pattern_full(input_grid: list[int]) -> list[int]:
    size = len(input_grid)

    # 1. Identify the correct pattern size P by finding the largest P ≥ 2
    #    such that input[0:P] == input[P:2P].
    max_half = size // 2
    pattern_size = None
    for P in range(max_half, 1, -1):
        if input_grid[:P] == input_grid[P:2 * P]:
            pattern_size = P
            break
    if pattern_size is None:
        # No valid repeated pattern found; return copy
        return input_grid.copy()

    # 2. Extract the pattern
    pattern = input_grid[:pattern_size]

    # 3. Build the output by tiling the pattern
    output = [0] * size
    pos = 0
    # Full tiles
    while pos + pattern_size <= size:
        output[pos:pos + pattern_size] = pattern
        pos += pattern_size
    # Partial tile for any remainder
    for i in range(pos, size):
        output[i] = pattern[i - pos]

    return output


def task_gravity_weighted_colors(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where color 2 is heavier than color 1 in gravity."""
    # Generate random field with only colors 1 and 2
    question = [rng.randint(1, 2) if rng.random() < 0.5 else 0 for _ in range(size)]

    # Count colors
    count_1 = sum(1 for x in question if x == 1)
    count_2 = sum(1 for x in question if x == 2)

    # Create answer with sorted colors
    answer = gen_field(size)

    # Place heavier color 2 first
    for i in range(count_2):
        answer[i] = 2

    # Then place color 1
    for i in range(count_1):
        answer[count_2 + i] = 1

    return {"input": question, "output": answer}


def transform_gravity_weighted_colors(input_grid: list[int], direction: str = "right") -> list[int]:
    size = len(input_grid)
    # Count occurrences
    count1 = sum(1 for x in input_grid if x == 1)
    count2 = sum(1 for x in input_grid if x == 2)

    output = [0] * size

    if direction == "right":
        # Place all 2's first, then 1's
        for i in range(count2):
            output[i] = 2
        for i in range(count1):
            output[count2 + i] = 1

    elif direction == "left":
        # Place all 2's at the right end, then 1's before them
        for i in range(count2):
            output[size - 1 - i] = 2
        for i in range(count1):
            output[size - 1 - count2 - i] = 1

    else:
        raise ValueError("direction must be 'right' or 'left'")

    return output



def task_color_left_half_blocks(rng: Random, size: int) -> Optional[dict[str, list[int]]]:
    """Generate a task where left half of blocks are colored differently."""
    pos = 0
    question = gen_field(size)
    blocks = []

    # Generate blocks with gap 1
    while pos < size:
        if rng.random() < 0.4:
            block_size = rng.randint(2, size // 2)
            if pos + block_size > size:
                break

            blocks.append((pos, block_size))
            for i in range(block_size):
                question[pos + i] = 2
            pos += block_size + 1  # block size + gap
        else:
            pos += 1

    if len(blocks) < 2:
        return None

    # Create answer with half-colored blocks
    answer = question.copy()
    for pos, block_size in blocks:
        half_size = block_size // 2
        for i in range(half_size):
            answer[pos + i] = 8

    return {"input": question, "output": answer}


def transform_color_left_half_blocks(input_grid: list[int]) -> list[int]:
    size = len(input_grid)
    output = input_grid.copy()
    i = 0

    while i < size:
        # detect start of a block of 2’s
        if input_grid[i] == 2:
            start = i
            # advance to end of this block
            while i < size and input_grid[i] == 2:
                i += 1
            block_len = i - start
            # only blocks of length ≥ 2 were generated, but we guard anyway
            if block_len >= 2:
                half = block_len // 2
                for offset in range(half):
                    output[start + offset] = 8
        else:
            i += 1

    return output


def task_mirror(task_result: Optional[dict[str, list[int]]]) -> Optional[dict[str, list[int]]]:
    """Mirror the input and output arrays of a task result."""
    if task_result is None:
        return None
    return {"input": list(reversed(task_result["input"])), "output": list(reversed(task_result["output"]))}


def task_inverse(task_result: Optional[dict[str, list[int]]]) -> Optional[dict[str, list[int]]]:
    """Swap the input and output arrays of a task result."""
    if task_result is None:
        return None
    return {"input": task_result["output"], "output": task_result["input"]}


def task_identity(task_result: Optional[dict[str, list[int]]]) -> Optional[dict[str, list[int]]]:
    """Return the task result unchanged."""
    return task_result


# Table of all ARC 1D task functions with their parameters
ARC_1D_TASKS = {
    # Move tasks - right direction
    "move_1pix_solid_right": (task_move_n_pix, {"move_pix": 1, "solid": True}, transform_move_n_pix, {"move_pix": 1, "direction": "right"}, "Let's try move all pixels to the right by 1 pixel"),
    "move_2pix_solid_right": (task_move_n_pix, {"move_pix": 2, "solid": True}, transform_move_n_pix, {"move_pix": 2, "direction": "right"}, "Let's try move all pixels to the right by 2 pixels"),
    "move_3pix_solid_right": (task_move_n_pix, {"move_pix": 3, "solid": True}, transform_move_n_pix, {"move_pix": 3, "direction": "right"}, "Let's try move all pixels to the right by 3 pixels"),
    "move_4pix_solid_right": (task_move_n_pix, {"move_pix": 4, "solid": True}, transform_move_n_pix, {"move_pix": 4, "direction": "right"}, "Let's try move all pixels to the right by 4 pixels"),
    "move_1pix_colorful_right": (task_move_n_pix, {"move_pix": 1, "solid": False}, transform_move_n_pix, {"move_pix": 1, "direction": "right"}, "Let's try move all pixels to the right by 1 pixel"),
    "move_2pix_colorful_right": (task_move_n_pix, {"move_pix": 2, "solid": False}, transform_move_n_pix, {"move_pix": 2, "direction": "right"}, "Let's try move all pixels to the right by 2 pixels"),
    "move_3pix_colorful_right": (task_move_n_pix, {"move_pix": 3, "solid": False}, transform_move_n_pix, {"move_pix": 3, "direction": "right"}, "Let's try move all pixels to the right by 3 pixels"),
    "move_4pix_colorful_right": (task_move_n_pix, {"move_pix": 4, "solid": False}, transform_move_n_pix, {"move_pix": 4, "direction": "right"}, "Let's try move all pixels to the right by 4 pixels"),
    # Move tasks - left direction (mirrored)
    "move_1pix_solid_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 1, "solid": True},
        transform_move_n_pix,
        {"move_pix": 1, "direction": "left"},
        "Let's try move all pixels to the left by 1 pixel",
    ),
    "move_2pix_solid_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 2, "solid": True},
        transform_move_n_pix,
        {"move_pix": 2, "direction": "left"},
        "Let's try move all pixels to the left by 2 pixels",
    ),
    "move_3pix_solid_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 3, "solid": True},
        transform_move_n_pix,
        {"move_pix": 3, "direction": "left"},
        "Let's try move all pixels to the left by 3 pixels",
    ),
    "move_4pix_solid_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 4, "solid": True},
        transform_move_n_pix,
        {"move_pix": 4, "direction": "left"},
        "Let's try move all pixels to the left by 4 pixels",
    ),
    "move_1pix_colorful_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 1, "solid": False},
        transform_move_n_pix,
        {"move_pix": 1, "direction": "left"},
        "Let's try move all pixels to the left by 1 pixel",
    ),
    "move_2pix_colorful_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 2, "solid": False},
        transform_move_n_pix,
        {"move_pix": 2, "direction": "left"},
        "Let's try move all pixels to the left by 2 pixels",
    ),
    "move_3pix_colorful_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 3, "solid": False},
        transform_move_n_pix,
        {"move_pix": 3, "direction": "left"},
        "Let's try move all pixels to the left by 3 pixels",
    ),
    "move_4pix_colorful_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 4, "solid": False},
        transform_move_n_pix,
        {"move_pix": 4, "direction": "left"},
        "Let's try move all pixels to the left by 4 pixels",
    ),
    # Move wrapped tasks - right direction
    "move_1pix_solid_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 1, "solid": True}, transform_move_n_pix_wrapped, {"move_pix": 1, "direction": "right"}, "Let's try move all pixels to the right by 1 pixel, and if it goes out of the array, wrap around the left of the array"),
    "move_2pix_solid_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 2, "solid": True}, transform_move_n_pix_wrapped, {"move_pix": 2, "direction": "right"}, "Let's try move all pixels to the right by 2 pixels, and if it goes out of the array, wrap around the left of the array"),
    "move_3pix_solid_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 3, "solid": True}, transform_move_n_pix_wrapped, {"move_pix": 3, "direction": "right"}, "Let's try move all pixels to the right by 3 pixels, and if it goes out of the array, wrap around the left of the array"),
    "move_4pix_solid_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 4, "solid": True}, transform_move_n_pix_wrapped, {"move_pix": 4, "direction": "right"}, "Let's try move all pixels to the right by 4 pixels, and if it goes out of the array, wrap around the left of the array"),
    "move_1pix_colorful_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 1, "solid": False}, transform_move_n_pix_wrapped, {"move_pix": 1, "direction": "right"}, "Let's try move all pixels to the right by 1 pixel, and if it goes out of the array, wrap around the left of the array"),
    "move_2pix_colorful_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 2, "solid": False}, transform_move_n_pix_wrapped, {"move_pix": 2, "direction": "right"}, "Let's try move all pixels to the right by 2 pixels, and if it goes out of the array, wrap around the left of the array"),
    "move_3pix_colorful_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 3, "solid": False}, transform_move_n_pix_wrapped, {"move_pix": 3, "direction": "right"}, "Let's try move all pixels to the right by 3 pixels, and if it goes out of the array, wrap around the left of the array"),
    "move_4pix_colorful_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 4, "solid": False}, transform_move_n_pix_wrapped, {"move_pix": 4, "direction": "right"}, "Let's try move all pixels to the right by 4 pixels, and if it goes out of the array, wrap around the left of the array"),
    # Move wrapped tasks - left direction (mirrored)
    "move_1pix_solid_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 1, "solid": True},
        transform_move_n_pix_wrapped,
        {"move_pix": 1, "direction": "left"},
        "Let's try move all pixels to the left by 1 pixel, and if it goes out of the array, wrap around the right of the array",
    ),
    "move_2pix_solid_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 2, "solid": True},
        transform_move_n_pix_wrapped,
        {"move_pix": 2, "direction": "left"},
        "Let's try move all pixels to the left by 2 pixels, and if it goes out of the array, wrap around the right of the array",
    ),
    "move_3pix_solid_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 3, "solid": True},
        transform_move_n_pix_wrapped,
        {"move_pix": 3, "direction": "left"},
        "Let's try move all pixels to the left by 3 pixels, and if it goes out of the array, wrap around the right of the array",
    ),
    "move_4pix_solid_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 4, "solid": True},
        transform_move_n_pix_wrapped,
        {"move_pix": 4, "direction": "left"},
        "Let's try move all pixels to the left by 4 pixels, and if it goes out of the array, wrap around the right of the array",
    ),
    "move_1pix_colorful_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 1, "solid": False},
        transform_move_n_pix_wrapped,
        {"move_pix": 1, "direction": "left"},
        "Let's try move all pixels to the left by 1 pixel, and if it goes out of the array, wrap around the right of the array",
    ),
    "move_2pix_colorful_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 2, "solid": False},
        transform_move_n_pix_wrapped,
        {"move_pix": 2, "direction": "left"},
        "Let's try move all pixels to the left by 2 pixels, and if it goes out of the array, wrap around the right of the array",
    ),
    "move_3pix_colorful_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 3, "solid": False},
        transform_move_n_pix_wrapped,
        {"move_pix": 3, "direction": "left"},
        "Let's try move all pixels to the left by 3 pixels, and if it goes out of the array, wrap around the right of the array",
    ),
    "move_4pix_colorful_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 4, "solid": False},
        transform_move_n_pix_wrapped,
        {"move_pix": 4, "direction": "left"},
        "Let's try move all pixels to the left by 4 pixels, and if it goes out of the array, wrap around the right of the array",
    ),
    # Gravity tasks - right direction
    "gravity_right": (task_gravity, {}, transform_gravity, {"direction": "right"}, "Let's first find all the zero elements in the array. Then let's move them to the right, while moving all the non-zero elements to the left."),
    "gravity_counting_right": (task_gravity_counting, {}, transform_gravity_counting, {"direction": "right"}, "Let’s first count how many non-zero elements are present in the input array. Then, we replace those non-zero elements with the number 1, keeping the same count. All remaining positions are filled with 0s, pushing all the 1s to the left."),
    "gravity_antigravity_right": (task_gravity_antigravity, {}, transform_gravity_antigravity, {"direction": "right"}, "Let’s first separate all the 1s and 2s in the array. All the 2s are “heavier” and move to the left, while all the 1s are “lighter” and move to the right. Let's fill the remaining space in the middle with 0s."),
    "gravity_one_step_right": (task_gravity_one_step, {}, transform_gravity_one_step, {"direction": "right"}, "Let’s move each non-zero element one step to the left if the space to its left is 0. Let's do this in order from left to right, so that only one step is allowed per element, and each element only moves if the immediate position on its left is unoccupied."),
    "gravity_weighted_colors_right": (task_gravity_weighted_colors, {}, transform_gravity_weighted_colors, {"direction": "right"}, "Let’s count how many 2s and how many 1s are in the array. Since color 2 is heavier than color 1, we move all the 2s to the leftmost positions first. After that, we place all the 1s to the right of the 2s. The remaining positions are filled with 0s."),
    # Gravity tasks - left direction (mirrored)
    "gravity_left": (lambda rng, size, **kwargs: task_mirror(task_gravity(rng, size, **kwargs)), {}, transform_gravity, {"direction": "left"}, "Let's first find all the zero elements in the array. Then let's move them to the left, while moving all the non-zero elements to the right."),
    "gravity_counting_left": (lambda rng, size, **kwargs: task_mirror(task_gravity_counting(rng, size, **kwargs)), {}, transform_gravity_counting, {"direction": "left"}, "Let’s first count how many non-zero elements are present in the input array. Then, we replace those non-zero elements with the number 1, keeping the same count. All remaining positions are filled with 0s, pushing all the 1s to the right."),
    "gravity_antigravity_left": (lambda rng, size, **kwargs: task_mirror(task_gravity_antigravity(rng, size, **kwargs)), {}, transform_gravity_antigravity, {"direction": "left"}, "Let’s first separate all the 1s and 2s in the array. All the 2s are “heavier” and move to the right, while all the 1s are “lighter” and move to the left. Let's fill the remaining space in the middle with 0s."),
    "gravity_one_step_left": (lambda rng, size, **kwargs: task_mirror(task_gravity_one_step(rng, size, **kwargs)), {}, transform_gravity_one_step, {"direction": "left"}, "Let’s move each non-zero element one step to the right if the space to its right is 0. Let's do this in order from right to left, so that only one step is allowed per element, and each element only moves if the immediate position on its right is unoccupied."),
    "gravity_weighted_colors_left": (
        lambda rng, size, **kwargs: task_mirror(task_gravity_weighted_colors(rng, size, **kwargs)),
        {},
        transform_gravity_weighted_colors,
        {"direction": "left"},
        "Let’s count how many 2s and how many 1s are in the array. Since color 2 is heavier than color 1, we move all the 2s to the rightmost positions first. After that, we place all the 1s to the left of the 2s. The remaining positions are filled with 0s."
    ),
    # Block tasks
    "block_touch_dot": (task_block_touch_dot, {}, transform_block_touch_dot, {}, "Let’s first find the position of the dot (value 1) and the block (a sequence of non-zero, non-1 values). Then let's try moving the block to directly touch the dot without overlapping it. If the block is to the left of the dot, we shift it right so its right end is adjacent to the dot. If the block is to the right of the dot, we shift it left so its left end touches the dot."),
    "block_touch_dot_1pix": (task_block_touch_dot_n_pix, {"move_pix": 1}, transform_block_touch_dot_n_pix, {"move_pix": 1}, "Let’s first locate the dot (color 2) and the block (a group of same nonzero values ≠ 2). Let's move the block 1 step closer to the dot but not past it. If the block is on the left of the dot, we move it right by up to 1 step or until it's adjacent to the dot. If it's on the right, we move it left similarly. The block’s shape and order stay unchanged, and the dot itself remains fixed in place."),
    "block_touch_dot_2pix": (task_block_touch_dot_n_pix, {"move_pix": 2}, transform_block_touch_dot_n_pix, {"move_pix": 2}, "Let’s first locate the dot (color 2) and the block (a group of same nonzero values ≠ 2). Let's move the block 2 steps closer to the dot but not past it. If the block is on the left of the dot, we move it right by up to 2 steps or until it's adjacent to the dot. If it's on the right, we move it left similarly. The block’s shape and order stay unchanged, and the dot itself remains fixed in place."),
    "block_touch_dot_3pix": (task_block_touch_dot_n_pix, {"move_pix": 3}, transform_block_touch_dot_n_pix, {"move_pix": 3}, "Let’s first locate the dot (color 2) and the block (a group of same nonzero values ≠ 2). Let's move the block 3 steps closer to the dot but not past it. If the block is on the left of the dot, we move it right by up to 3 steps or until it's adjacent to the dot. If it's on the right, we move it left similarly. The block’s shape and order stay unchanged, and the dot itself remains fixed in place."),
    "block_touch_dot_4pix": (task_block_touch_dot_n_pix, {"move_pix": 4}, transform_block_touch_dot_n_pix, {"move_pix": 4}, "Let’s first locate the dot (color 2) and the block (a group of same nonzero values ≠ 2). Let's move the block 4 steps closer to the dot but not past it. If the block is on the left of the dot, we move it right by up to 4 steps or until it's adjacent to the dot. If it's on the right, we move it left similarly. The block’s shape and order stay unchanged, and the dot itself remains fixed in place."),
    "block_scale_to_dot": (task_block_scale_to_dot, {}, transform_block_scale_to_dot, {}, "Let’s first locate the dot (color 2) and the block (a sequence of identical non-zero values different from 2). Let's try scale the block, so that it stretches just enough to touch the dot. Let's try to keep one end of the block fixed in its original position, and only the opposite end extends or contracts to reach the dot. If the block is on the left of the dot, it grows rightward. If it's on the right, it grows leftward. Let's try to keep the dot’s position unchanged."),
    "block_and_noise_remove": (task_block_and_noise_remove, {}, transform_block_and_noise_remove, {}, "Let’s first identify the main block — a group of consecutive cells with the same color. It looks like the block is surrounded by noise: individual pixels of the same color placed away from the block. Let's try remove all the pixels by setting them to 0."),
    "block_and_noise_remove_inside": (task_block_and_noise_remove_inside, {}, transform_block_and_noise_remove_inside, {}, "Let’s first find the main block — a long sequence of the same color. Looking at this block, some pixels may be corrupted by noise: values that are different from the block’s color. Let's try to clean up this block by replacing any mismatched values back to the correct block color, while leaving the rest of the array unchanged."),
    "move_block_by_own_size": (task_move_block_by_own_size, {}, transform_move_block_by_own_size, {}, "Let’s first identify a block of sequence of identical non-zero values. Let's move this block to the right by a distance equal to its own size."),
    # Pattern tasks
    "two_points_and_fill": (task_two_points_and_fill, {}, transform_two_points_and_fill, {"inverse": False}, "Let’s first find the two positions in the array that share the same non-zero color. Then let's try to fill the pixels between them all with that same color."),
    "two_points_and_fill_inv": (
        lambda rng, size, **kwargs: task_inverse(task_two_points_and_fill(rng, size, **kwargs)),
        {},
        transform_two_points_and_fill,
        {"inverse": True},
        "Let's first find the endpoints of a block of the same color. Then let's try to remove all colors by changing them to 0 in between the endpoints."
    ),
    "copy_block_to_dots": (task_copy_block_to_dots, {}, transform_copy_block_to_dots, {}, "Let’s start by identifying the original block pattern at the beginning of the array and the dot positions, which are marked with a single pixel of the same color. For every dot, let's insert a full copy of the block centered around it (with equal length to the original block), ensuring that the inserted blocks do not overlap and match the original pattern."),
    "copy_block_to_dots_colors": (task_copy_block_to_dots_colors, {}, transform_copy_block_to_dots_colors, {}, "Let’s try to identity the original block pattern at the start of the array, and then find each colored dot elsewhere in the array. For every dot, let's try to place a new block of the same size as the original, centered on the dot and filled entirely with the dot's color."),
    "repeat_pattern_full": (task_repeat_pattern_full, {}, transform_repeat_pattern_full, {}, "Let’s first extract the repeating pattern from the beginning of the array — the same sequence appears twice in a row. This defines the base pattern. Let's keep copying this pattern over and over until the entire array is filled. If there’s not enough space for a full repetition at the end, let's copy as many elements from the beginning of the pattern as will fit."),
    # Reflection tasks
    "reflect_block_with_border_pixel": (task_reflect_block_with_border_pixel, {}, transform_reflect_block_with_border_pixel, {}, "Let’s first find a block of sequence of repeated values with a distinct border pixel at either the left or right end. Let's reflects the entire block in place, so the leftmost and rightmost elements switch sides, and the order of all values in the block is flipped."),
    "reflect_block_random": (task_reflect_block_with_border_pixel_random, {}, transform_reflect_block_with_border_pixel_random, {}, "Let’s identify a block of a sequence of randomly colored values that includes a special border pixel marked with a distinct color at one end (either the left or the right). Let's reflect the entire block in place, reversing the order of its elements while keeping the block in the same position."),
    "reflect_block_around_dot": (task_reflect_block_around_dot, {}, transform_reflect_block_around_dot, {}, "Let's try to locate a dot of color 2 and a block of a sequence of same-colored values that appears entirely to the left or right of the dot. Let't reflect the block across the dot, as if the dot were a mirror."),
    # Color tasks
    "paint_biggest_block": (task_paint_biggest_block, {}, transform_paint_biggest_block, {}, "Let’s identify all the blocks of sequences of two or more adjacent cells with the same color that is not 0. Let's then find the largest block (the one with the most consecutive cells). If there’s only one block of that maximum size, let's try to repaint it with a new color 1, while keeping all other blocks unchanged."),
    "recolor_blocks_by_size": (task_recolor_blocks_by_size, {}, transform_recolor_blocks_by_size, {}, "Let’s first try to find the two blocks of sequences of color 3, separated by at least one zero. Let's try to compare the sizes, and pain recolor the larger block to color 1, and recolor the smaller block to 2."),
    "change_to_five": (task_change_to_five, {}, transform_change_to_five, {}, "Let’s go through the array and look for all non-zero values, and recolor them to 5."),
    "recolor_blocks_from_palette": (task_recolor_blocks_from_palette, {}, transform_recolor_blocks_from_palette, {}, "Let’s first extract the color palette at the beginning of the array — each value represents a distinct color. Then let's look for several blocks of color 5 of the same length. Let's recolor each block using the corresponding color in the color palette."),
    "color_left_half_blocks": (task_color_left_half_blocks, {}, transform_color_left_half_blocks, {}, "Let’s identify all the blocks of sequences of consecutive 2s separated by gaps. Then let's try to recolor the left half of each block to 8, and try to keep the right half unchanged."),
    # Sorting tasks
    "sort_blocks_by_size": (task_sort_blocks_by_size, {}, transform_sort_blocks_by_size, {}, "Let’s identify all the blocks of sequences of the same color separated by gaps of 0s. Let's sort the blocks by size in ascending order, and reassemble them from left to right in that order. Let's try to place one 0 in between the block."),
    "sort_complete_sequence": (task_sort_complete_sequence, {}, transform_sort_complete_sequence, {}, "Let’s identify all the blocks of sequences of the same color separated by gaps of 0s. Let's sort the blocks by size in ascending order, and reassemble them from left to right in that order. Let's try to place one 0 in between the block."),
    # Fill tasks
    "duplicate_block_from_seeds": (task_duplicate_block_from_seeds, {}, transform_duplicate_block_from_seeds, {}, "Let’s identify the main block of sequence of 1s, and one or two seed pixels placed some fixed distance away from the block. If the seed is on the left, let's replicate the block leftward toward the array’s start, repeating it with gaps until it hits the edge. If the seed is on the right, let's do the same but to the right."),
    "fill_from_pixel": (task_fill_from_pixel, {}, transform_fill_from_pixel, {}, "Let's first try to identify a single colored value placed either directly before or after a larger block of different color, and that is the seed pixel. If the seed is on the left of the block, let's spread the seed color leftward from the seed position to the beginning of the array. If the seed is on the right, it fills rightward to the end."),
    "fill_until_collision": (task_fill_until_collision, {}, transform_fill_until_collision, {}, "Let’s locate the side marker (5), which determines the direction of filling: left-to-right if at the start, or right-to-left if at the end. We also find several colored pixels scattered in the array. These pixels act as color sources, and each one fills the empty (0) space from the previous fill boundary up to its position. The fill proceeds in order — either left-to-right or right-to-left — and each region between color sources is filled with the current pixel’s color, stopping just before the next one."),
    # Marking tasks
    "mark_size_two_blocks": (task_mark_size_two_blocks, {}, transform_mark_size_two_blocks, {}, "If a block has exactly size 2, we mark it by placing a special marker color (3) in the cells immediately before and after it (if within bounds)."),
}
