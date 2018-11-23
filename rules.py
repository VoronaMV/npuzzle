import math
import numpy as np
from state import State


class Rule:

    HEURISTICS_CHOICES = {
        "H": "simple",
        "M": "manhattan",
        "D": "diagonal",
        "E": "euclidean",
        "ML": "manhattan_linear" # TODO: Implement it
    }
    heuristics = None

    class WrongHeuristicsError(Exception):

        def __init__(self, message='Invalid heuristics. Available are {}', error=None):
            availableheuristics = ' '.join(Rule.HEURISTICS_CHOICES or [])
            formatted_message = message.format(availableheuristics)
            super().__init__(formatted_message)
            self.error = error

    @classmethod
    def choose_heuristics(cls, heuristic_name: str) -> None:
        if heuristic_name not in cls.HEURISTICS_CHOICES:
            raise cls.WrongHeuristicsError()
        prefix = 'heuristic_'
        default_heuristic = prefix + cls.HEURISTICS_CHOICES.get(heuristic_name)
        cls.heuristics = cls.__dict__.get(default_heuristic, cls.__dict__.get(default_heuristic))

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
