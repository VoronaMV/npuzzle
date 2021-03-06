import math
from state import State
import numpy as np

HEURISTICS_CHOICES = {
    "H": "simple",
    "M": "manhattan",
    "D": "diagonal",
    "E": "euclidean",
    "ML": "manhattan_linear"  # TODO: Implement it
}
heuristics = None


class WrongHeuristicsError(Exception):

    def __init__(self, message='Invalid heuristics. Available are {}', error=None):
        availableheuristics = ' '.join(HEURISTICS_CHOICES or [])
        formatted_message = message.format(availableheuristics)
        super().__init__(formatted_message)
        self.error = error


class Heuristics:
    def get_total_h(self, node: State) -> int:
        raise NotImplementedError()


class Manhattan(Heuristics):

    def get_total_h(self, node: State) -> int:
        """
        Returns manhattan distance of all puzzles compare with terminal state
        abs(cur(i,j) - target(i,j))
        """
        manhattan_sum = 0
        for indx_pair, value in np.ndenumerate(node._map):
            # Compare with indexes of terminal state
            for t_indx_pair, t_value in np.ndenumerate(node.terminal_map):
                if value == t_value:
                    diff = np.subtract(indx_pair, t_indx_pair)
                    abs_diff = abs(diff)
                    manhattan_sum += sum(abs_diff)
                    break
        return manhattan_sum


class ManhattanLinear(Heuristics):


    @staticmethod
    def get_row_conflict(terminal_map, current_map, row):
        conflict_sum = 0
        current_map_row = current_map[row, :]
        terminal_map_row = terminal_map[row, :]
        intersection_arr = np.intersect1d(current_map_row, terminal_map_row)
        for i, elem in enumerate(current_map_row):
            for puzzle in current_map_row[i:]:
                if elem in intersection_arr and elem > puzzle:
                    conflict_sum += 1

        return conflict_sum

    @staticmethod
    def get_col_conflict(terminal_map, current_map, col):
        conflict_sum = 0
        current_map_col = current_map[:, col]
        terminal_map_col = terminal_map[:, col]
        intersection_arr = np.intersect1d(current_map_col, terminal_map_col)
        for i, elem in enumerate(current_map_col):
            for puzzle in current_map_col[i:]:
                if elem in intersection_arr and elem > puzzle:
                    conflict_sum += 1
        return conflict_sum

    def get_total_h(self, node: State) -> int:
        conflict_sum = 0
        manhattan_sum = 0
        check_row = -1
        check_col = -1
        for indx_pair, value in np.ndenumerate(node._map):

            row, col = indx_pair
            if check_row != row:
                check_row = row
                conflict_sum += self.get_row_conflict(node.terminal_map, node._map, row)
                conflict_sum += self.get_col_conflict(node.terminal_map, node._map, row)

            # Compare with indexes of terminal state
            for t_indx_pair, t_value in np.ndenumerate(node.terminal_map):
                # if self.check_linear_conflict(value, t_value, indx_pair, t_indx_pair):
                #     conflict_sum += 1
                if value == t_value:
                    diff = np.subtract(indx_pair, t_indx_pair)
                    abs_diff = abs(diff)
                    manhattan_sum += sum(abs_diff)
                    break
        # print('CON_SUM', conflict_sum)
        return manhattan_sum + (2 * conflict_sum)


class Hemming(Heuristics):

    def get_total_h(self, node: State) -> int:
        """
        Returns a number of puzzles at wrong place.
        """
        eq_array = np.equal(node._map, node.terminal_map)
        wrong_placed_puzzles = len(eq_array[eq_array == False])
        return wrong_placed_puzzles


class Euclidean(Heuristics):

    def get_total_h(self, node: State) -> int:
        total_sum = 0
        for indx_pair, value in np.ndenumerate(node._map):
            for t_indx_pair, t_value in np.ndenumerate(node.terminal_map):
                if value == t_value:
                    diff = np.subtract(indx_pair, t_indx_pair)
                    abs_diff = abs(diff)
                    total_sum += math.sqrt(abs_diff[0] ** 2 + abs_diff[1] ** 2)
                    break
        return total_sum


class Diagonal(Heuristics):

    def get_total_h(self, node: State) -> int:
        total_sum = 0
        for indx_pair, value in np.ndenumerate(node._map):
            for t_indx_pair, t_value in np.ndenumerate(node.terminal_map):
                if value == t_value:
                    diff = np.subtract(indx_pair, t_indx_pair)
                    abs_diff = abs(diff)
                    total_sum += sum(abs_diff) + (math.sqrt(2) - 2) * min(abs_diff)
                    break
        return total_sum


class Unicost(Heuristics):
    """Heuristics for Uniform-cost search (h is always 0)"""
    def get_total_h(self, node: State) -> int:
        return 0


class Heuristic:

    def get_heuristic(self, name, unicost):
        raise NotImplementedError()


class PuzzleHeuristic(Heuristic):

    def get_heuristic(self, name, unicost):
        if unicost:
            return Unicost()
        if name == 'M':
            return Manhattan()
        elif name == 'ML':
            return ManhattanLinear()
        elif name == 'H':
            return Hemming()
        elif name == 'E':
            return Euclidean()
        elif name == 'D':
            return Diagonal()
        else:
            raise WrongHeuristicsError()
