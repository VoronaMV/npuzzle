import os
import re
import sys
import datetime
import numpy as np
from typing import Deque


class NPuzzlesMap:

    MIN_SHAPE = (3, 3)

    def __init__(self, shape: tuple, initial_map: np.ndarray):
        if shape < self.MIN_SHAPE:
            raise self.BadMapError()
        self.initial_map = initial_map
        self.initial_state = State(self.initial_map)
        dimension, _ = shape
        terminal_flat_array = np.append(np.arange(1, dimension**2), 0)
        terminal_array = np.reshape(terminal_flat_array, shape)
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

    @classmethod
    def from_file(cls, filename):
        initial_map = cls.__map_from_file(filename)
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

    def __init__(self, data: np.ndarray, parent=None, ):
        if not isinstance(data, np.ndarray) and data.size < 9:
            raise State.BadMapError()
        self._map = data.astype(int)
        self.flat_map = self._map.flatten()
        self.parent = parent
        self.g = parent.g + 1 if parent else 0
        self.f = None
        # TODO: Make better way
        if self.terminal_map is None:
            dimension, _ = self._map.shape
            terminal_flat_array = np.append(np.arange(1, dimension ** 2), 0)
            self.terminal_map = np.reshape(terminal_flat_array, self._map.shape)

        self.empty_puzzle_coord = self.empty_element_coordinates(self._map)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise self.UnknownInstanceError()
        return np.array_equal(self._map, other._map)

    def __str__(self):
        return str(self._map)

    def __repr__(self):
        return self.__str__()

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

    @staticmethod
    def empty_element_coordinates(_map: np.ndarray) -> tuple:
        for indx_pair, elem in np.ndenumerate(_map):
            if elem == 0:
                return indx_pair


# TODO: Do we need Dequee?
class TState(Deque):

    def __init__(self, *args, **kwargs):
        self.appends_amount = 0
        super().__init__(*args, **kwargs)

    @property
    def time_complexity(self):
        return self.appends_amount

    def append(self, item):
        self.appends_amount += 1
        return super().append(item)

    def find_min_state(self, heuristic: callable) -> State:
        min_state = self[0]
        if not min_state.f:
            min_state.f = min_state.g + heuristic(min_state)

        for elem in self:
            if not elem.f:
                elem.f = elem.g + heuristic(elem)

            if elem.f < min_state.f:
                min_state = elem
                min_state.f = elem.f
        return min_state

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

    HEURISTICS_CHOICES = ('simple', 'manhattan', )
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
        default_heuristic = prefix + cls.HEURISTICS_CHOICES[0]
        cls._heuristics = cls.__dict__.get(prefix + heuristic_name, default_heuristic)

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
    def neignbours(node: State) -> list:
        """
        Should return set of app states that can be neighbours to current
        map state (shift 0 right/left/up/down)
        """
        neighbours = list()
        directions = ['up', 'down', 'right', 'left']
        for direction in directions:
            coordinates = node.shift_empty_puzzle(direction)
            try:
                if coordinates[0] < 0 and coordinates[1] < 0:
                    raise Exception()
                # TODO: switch elements
                new_map = node._map.copy()
                non_empty_element = node._map[coordinates]
                new_map[node.empty_puzzle_coord] = non_empty_element
                new_map[coordinates] = 0
                neighbours.append(State(new_map, parent=node))
            except:
                continue
        return neighbours

    @staticmethod
    def distance(first: State, second: State) -> float:
        """
        This is a simple case. So distance between each state is 1
        :return: 1.0
        """
        # raise NotImplementedError()
        return 1.0


def get_size_comlexity(_open, _close, *args):
    return len(_open) + len(_close) + len(args)

if __name__ == '__main__':

    start_time = datetime.datetime.now()

    heuristics_name = sys.argv[1].lower()
    Rule.choose_heuristics(heuristics_name)

    # npazzle = NPuzzlesMap.from_file('4_4_map.txt')
    npazzle = NPuzzlesMap.from_file('3_3_map_test.txt')

    initial_state = npazzle.initial_state
    terminal_state = npazzle.terminal_state

    _open = TState()
    _close = TState()
    _open.append(initial_state)

    # TODO: Check for validity. It's only assumption
    size_comlexity = get_size_comlexity(_open, _close)

    while _open:
        min_state = _open.find_min_state(Rule._heuristics)
        if min_state == terminal_state:
            solution = TState(elem for elem in _open.reverse_to_head(min_state))
            solution.reverse()  # now it is solution
            moves_number = len(solution)
            end_time = datetime.datetime.now()
            delta = end_time - start_time
            print('seconds: ', delta.seconds)
            print(f'size complexity: {size_comlexity}')
            print(f'time complexity: {_open.time_complexity}')
            print(f'Moves: {moves_number}')
            # print(globals().keys())
            exit(str(solution))
        _open.remove(min_state)
        _close.append(min_state)

        neighbours = Rule.neignbours(min_state)

        tmp_size = get_size_comlexity(_open, _close, *neighbours)
        if size_comlexity < tmp_size:
            size_comlexity = tmp_size

        for neighbour in neighbours:
            g = min_state.g + Rule.distance(min_state, neighbour)

            if neighbour in _close:
                continue
            is_g_better = False

            if neighbour not in _open:
                _open.append(neighbour)
                is_g_better = True
            else:
                is_g_better = g < neighbour.g

            if is_g_better:
                neighbour.parent = min_state
                neighbour.g = g
