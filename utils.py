import numpy as np
import argparse


TERMINAL_STATES = {
    "ordinary": {
        3: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]]),
        4: np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]),
        5: ""},
    "snail": {
        3: np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]]),
        4: np.array([[1, 2, 3, 4], [12, 13, 14, 5], [11, 0, 15, 6], [10, 9, 8, 7]]),
        5: ""}
}

# TERMINAL_STATES = {
#         # 3: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]]),
#         # 4: np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]),
#         3: np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]]),
#         4: np.array([[1, 2, 3, 4], [12, 13, 14, 5], [11, 0, 15, 6], [10, 9, 8, 7]]),
# }


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


def argument_parser():
    generator = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    generator.add_argument('-G', '--generate', type=int, help="Size of the puzzle's side. Must be >= 3.", default=3,
                           dest='size')
    generator.add_argument("-s", "--solvable", action="store_true", default=False,
                           help="Forces generation of a solvable puzzle. Overrides -u.")
    generator.add_argument("-u", "--unsolvable", action="store_true", default=False,
                           help="Forces generation of an unsolvable puzzle")
    generator.add_argument("-i", "--iterations", type=int, default=10000, help="Number of passes")

    parser = argparse.ArgumentParser(parents=[generator], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--file', default='', type=str, help='Enter a path to file with puzzle',)
    parser.add_argument('-H', '--heuristics', choices=['M', 'ML', 'H', 'E', 'D'], default='M',
                        dest='heuristics',
                        help='''Choose one of heuristics to solve the puzzle.
M - for Manhattan distance.
ML - for Manhattan distance + Linear conflict.
H - for Hemming distance.
E - for Euclidean distance.
D - for Diagonal distance.
Default value is M''')

    parser.add_argument('-q', '--queuesize', type=int, default=8, help='''Here you can set the size of the Queue.
The bigger size = the shorter way, but longer time.
Default value is 8''', dest='q_size')
    parser.add_argument('-O', '--ordinary', help='Changes the terminal state to Ordinary. Default is Snail',
                        action='store_true')

    args = parser.parse_args()
    print(args)
    return args
