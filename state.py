import os
import re
import math
import time
import argparse
import numpy as np
from generator import generate_puzzle
from hashlib import sha1
from typing import Deque
from queue import PriorityQueue


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


class NPuzzlesMap:

    MIN_SHAPE = (3, 3)

    def __init__(self, shape: tuple, initial_map: np.ndarray):
        if shape < self.MIN_SHAPE:
            raise self.BadMapError()
        self.initial_map = initial_map
        self.initial_state = State(self.initial_map)
        dimension, _ = shape
        terminal_array = TERMINAL_STATES.get(dimension)
        self.terminal_state = State(terminal_array)

    @staticmethod
    def __map_from_file(filename: str) -> np.ndarray:
        dimension = None
        start_map = []
        if not os.path.isfile(filename):
            raise Exception(f'File {filename} does not exist')
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if dimension is None and re.fullmatch(r'\d+', line):
                    dimension = int(line)
                elif dimension is not None:
                    row = re.findall(r'\d+', line)
                    row = [int(digit) for digit in row[:dimension]]
                    start_map.append(row)
        return np.array(start_map)

    @staticmethod
    def __map_from_string(string_map: str) -> np.ndarray:
        dimension = None
        start_map = list()
        lines = string_map.split('\n')
        lines.pop(-1)
        for line in lines:
            line.strip()
            if dimension is None and re.fullmatch(r'\d+', line):
                dimension = int(line)
            elif dimension is not None:
                row = re.findall(r'\d+', line)
                row = [int(digit) for digit in row[:dimension]]
                start_map.append(row)
        return np.array(start_map)

    @classmethod
    def from_file(cls, filename):
        initial_map = cls.__map_from_file(filename)
        return cls(initial_map.shape, initial_map)

    @classmethod
    def from_string(cls, string_map):
        initial_map = cls.__map_from_string(string_map)
        return cls(initial_map.shape, initial_map)

    def __str__(self):
        return f'Initial{self.initial_state}, Terminal{self.terminal_state}'

    def __repr__(self):
        return self.__str__()

    class BadMapError(Exception):
        def __init__(self, message='Bad map', error=None):
            super().__init__(message)
            self.error = error


class State:

    terminal_map = None

    def __init__(self, data: np.ndarray, parent=None, heuristic: callable=None):
        if not isinstance(data, np.ndarray) and data.size < 9:
            raise State.BadMapError()
        self._map = data.astype(int)
        self.flat_map = self._map.flatten()
        self.parent = parent
        self.g = parent.g + 1 if parent else 0
        # TODO: Make better way
        if self.terminal_map is None:
            dimension, _ = self._map.shape
            self.terminal_map = TERMINAL_STATES.get(dimension)

        self.empty_puzzle_coord = self.empty_element_coordinates(self._map)
        if heuristic:
            self.f = g + heuristic(self)
        else:
            self.f = None
        self.hash = sha1(self._map).hexdigest()

    def __eq__(self, other):
        return self.hash == other.hash

    def __str__(self):
        return str(self._map)

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self.f < other.f

    def __le__(self, other):
        return self.f <= other.f

    def __gt__(self, other):
        return self.f > other.f

    def __ge__(self, other):
        return self.f >= other.f

    class UnknownInstanceError(Exception):
        def __init__(self, message='Unknown class instance', error=None):
            super().__init__(message)
            self.error = error

    class BadMapError(Exception):
        def __init__(self, message='Bad map', error=None):
            super().__init__(message)
            self.error = error

    def shift_empty_puzzle(self, direction):
        row_indx, col_indx = self.empty_puzzle_coord
        directions = {
            'up': (row_indx - 1, col_indx),
            'down': (row_indx + 1, col_indx),
            'right': (row_indx, col_indx + 1),
            'left': (row_indx, col_indx - 1)
        }
        try:
            new_coords = directions[direction]
        except KeyError:
            raise Exception(f'Wrong Direction. Available are: {direction.keys()}')
        else:
            return new_coords

    def set_metrics(self, heuristic: callable, g=None):
        # TODO: Make exception for no heuristics
        if g:
            self.g = g
        self.f = self.g + heuristic(self)

    @staticmethod
    def empty_element_coordinates(_map: np.ndarray) -> tuple:
        for indx_pair, elem in np.ndenumerate(_map):
            if elem == 0:
                return indx_pair


class TState(PriorityQueue):

    def __contains__(self, item: State) -> bool:
        matches = (True for state in self.queue if item == state)
        return next(matches, False)

    def __str__(self):
        # TODO: Change it
        res = ''
        for elem in self.queue:
            res += str(elem) + '\n\n'
        return res

    def put_nowait(self, item):
        if self.maxsize and self.maxsize == self.qsize():
            self.pop_max()
        return super().put_nowait(item)

    def pop_max(self):
        max_element = max(self.queue)
        if not max_element:
            return
        max_index = self.queue.index(max_element)
        return self.queue.pop(max_index)


class TStateDeque(Deque):

    def __init__(self, *args, **kwargs):
        self.appends_amount = 0
        super().__init__(*args, **kwargs)

    @property
    def time_complexity(self):
        return self.appends_amount

    def append(self, item):
        self.appends_amount += 1
        return super().append(item)

    @staticmethod
    def reverse_to_head(state: State) -> iter:
        while state:
            yield state
            state = state.parent

    def __contains__(self, item: State) -> bool:
        matches = (True for state in self if item == state)
        return next(matches, False)

    def __str__(self):
        # TODO: Change it
        res = ''
        for elem in self:
            res += str(elem) + '\n\n'
        return res


class Rule:

    HEURISTICS_CHOICES = {
        "H": "simple",
        "M": "manhattan",
        "D": "diagonal",
        "E": "euclidean",
        "ML": "manhattan_linear" # TODO: Implement it
    }
    _heuristics = None

    class WrongHeuristicsError(Exception):

        def __init__(self, message='Invalid heuristics. Available are {}', error=None):
            available_heuristics = ' '.join(Rule.HEURISTICS_CHOICES or [])
            formatted_message = message.format(available_heuristics)
            super().__init__(formatted_message)
            self.error = error

    @classmethod
    def choose_heuristics(cls, heuristic_name: str) -> None:
        if heuristic_name not in cls.HEURISTICS_CHOICES:
            raise cls.WrongHeuristicsError()
        prefix = 'heuristic_'
        default_heuristic = prefix + cls.HEURISTICS_CHOICES.get(heuristic_name)
        cls._heuristics = cls.__dict__.get(default_heuristic, cls.__dict__.get(default_heuristic))

    @staticmethod
    def heuristic_simple(node: State) -> int:
        """
        Returns a number of puzzles at wrong place.
        """
        eq_array = np.equal(node._map, node.terminal_map)
        wrong_placed_puzzles = len(eq_array[eq_array == False])
        return wrong_placed_puzzles

    @staticmethod
    def heuristic_manhattan(node: State) -> int:
        """
        Returns manhattan distance of all puzzles compare with terminal state
        abs(cur(i,j) - target(i,j))
        """
        total_sum = 0
        for indx_pair, value in np.ndenumerate(node._map):
            # Compare with indexes of terminal state
            for t_indx_pair, t_value in np.ndenumerate(node.terminal_map):
                if value == t_value:
                    diff = np.subtract(indx_pair, t_indx_pair)
                    abs_diff = abs(diff)
                    total_sum += sum(abs_diff)
                    break
        return total_sum

    @staticmethod
    def heuristic_diagonal(node: State) -> int:
        total_sum = 0
        for indx_pair, value in np.ndenumerate(node._map):
            for t_indx_pair, t_value in np.ndenumerate(node.terminal_map):
                if value == t_value:
                    diff = np.subtract(indx_pair, t_indx_pair)
                    abs_diff = abs(diff)
                    total_sum += sum(abs_diff) + (math.sqrt(2) - 2) * min(abs_diff)
                    break
        return total_sum

    @staticmethod
    def heuristic_euclidean(node: State) -> int:
        total_sum = 0
        for indx_pair, value in np.ndenumerate(node._map):
            for t_indx_pair, t_value in np.ndenumerate(node.terminal_map):
                if value == t_value:
                    diff = np.subtract(indx_pair, t_indx_pair)
                    abs_diff = abs(diff)
                    total_sum += math.sqrt(abs_diff[0] ** 2 + abs_diff[1] ** 2)
                    break
        return total_sum

    @staticmethod
    def neighbours(node: State) -> list:
        """
        Should return set of app states that can be neighbours to current
        map state (shift 0 right/left/up/down)
        """
        _neighbours = list()
        directions = ['up', 'down', 'right', 'left']
        for direction in directions:
            coordinates = node.shift_empty_puzzle(direction)
            try:
                if coordinates[0] < 0 or coordinates[1] < 0:
                    continue
                new_map = node._map.copy()
                non_empty_element = node._map[coordinates]
                new_map[node.empty_puzzle_coord] = non_empty_element
                new_map[coordinates] = 0
                _neighbours.append(State(new_map, parent=node))
            except:
                continue
        return _neighbours

    @staticmethod
    def distance(first: State, second: State) -> float:
        """
        This is a simple case. So distance between each state is 1
        :return: 1.0
        """
        # raise NotImplementedError()
        return 1.0


if __name__ == '__main__':
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

    args = parser.parse_args()

    start_time = time.time()
    Rule.choose_heuristics(args.heuristics)
    if args.file:
        npazzle = NPuzzlesMap.from_file(filename=args.file)
    else:
        string_puzzle = generate_puzzle(args)
        npazzle = NPuzzlesMap.from_string(string_puzzle)

    initial_state = npazzle.initial_state
    initial_state.f = initial_state.g + Rule._heuristics(initial_state)

    terminal_state = npazzle.terminal_state

    # The bigger size = the shorter way, but longer time
    # 8 was very fast
    _open = TState(maxsize=8) # 50 was good for 4*4 3 на 3 тоже) Map 4*4 inversion=45 the best was max_size=15 (inv number=6 - time=0.4 sec)
    _close = TStateDeque()
    _open.put_nowait(initial_state)
    print(initial_state)
    print('solavble?', is_solvable(initial_state._map, dimension=initial_state._map.shape[1]))

    while not _open.empty():
        min_state = _open.get_nowait()

        if min_state == terminal_state:
            solution = TStateDeque(elem for elem in _close.reverse_to_head(min_state))
            solution.reverse()  # now it is solution
            moves_number = len(solution)
            end_time = time.time()
            delta = end_time - start_time
            print(str(solution))
            print('seconds: ', delta)
            print('open', _open.qsize())
            print(f'Moves: {moves_number}')
            # exit(str(solution))
            exit()

        _close.append(min_state)

        neighbours = Rule.neighbours(min_state)

        for neighbour in neighbours:
            if neighbour in _close:
                continue

            g = min_state.g + Rule.distance(min_state, neighbour)

            if neighbour not in _open:
                neighbour.parent = min_state
                neighbour.set_metrics(g=g, heuristic=Rule._heuristics)
                # neighbour.g = g
                # neighbour.f = neighbour.g + Rule._heuristics(neighbour)
                _open.put_nowait(neighbour)
            elif g <= neighbour.g:
                i = _open.queue.index(neighbour)
                neighbour = _open.queue[i]
                neighbour.parent = min_state
                neighbour.set_metrics(g=g, heuristic=Rule._heuristics)
                # neighbour.g = g
                # neighbour.f = neighbour.g + Rule._heuristics(neighbour)
