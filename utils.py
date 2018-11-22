import numpy as np


TERMINAL_STATES = {
    # 3: np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]]),
    4: np.array([[1, 2, 3, 4], [12, 13, 14, 5], [11, 0, 15, 6], [10, 9, 8, 7]]),
    3: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]]),
    # 4: np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])
}


def get_size_comlexity(_open, _close, *args):
    return len(_open) + len(_close) + len(args)


def is_solvable(_map: np.ndarray, dimension=4) -> bool:
    flat_map = _map.flatten()
    inversions = 0
    for i, puzzle in enumerate(flat_map):
        if puzzle == 0:
            continue
        for elem in flat_map[:i]:
            if elem > puzzle:
                inversions += 1
    print('inversions number=', inversions)
    is_inversions_even = True if inversions % 2 == 0 else False

    if dimension % 2 != 0:
        return is_inversions_even
    if 0 in _map[::-2]:
        # inversions even
        return is_inversions_even
    elif 0 in _map[::2]:
        # inversions odd
        return not is_inversions_even
    return False
