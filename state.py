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
        terminal_array = np.append(np.arange(1, dimension**2), 0)
        self.terminal_state = State(terminal_array)
        # State.terminal_state = self.terminal_state

    @classmethod
    def from_file(cls, filename):
        # TODO: Implement file reading here
        shape = (0, 0)
        initial_map = np.ndarray()
        cls(shape, initial_map)

    def __str__(self):
        return f'Initial{self.initial_state}, Terminal{self.terminal_state}'

    def __repr__(self):
        return  self.__str__()

    class BadMapError(Exception):
        def __init__(self, message='Bad map', error=None):
            super().__init__(message)
            self.error = error


class State:

    terminal_map = None

    def __init__(self, data: np.ndarray, parent=None, ):
        if not isinstance(data, np.ndarray) and data.size < 9:
            raise state.BadMapError()
        self._map = data.astype(int)
        self.flat_map = self._map.flatten()
        self.parent = parent
        self.g = parent.g + 1 if parent else 0
        # TODO: Make better way
        if self.terminal_map is None:
            dimension, _ = self._map.shape
            self.terminal_map = np.append(np.arange(1, dimension ** 2), 0)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise self.UnknownInstanceError()
        return np.array_equal(self._map, other._map)

    def __str__(self):
        return f'State(f={self.h}, g={self.g}, h={self.h})'

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

    @property
    def all_neighbours(self):
        """
        Should return set of app states that can be neighbours to current
        map state (shift 0 right/left/up/down)
        """
        raise NotImplementedError()

    @property
    def h(self) -> int:
        """
        Returns a number of puzzles at wrong place.
        """
        if not isinstance(self._map, np.ndarray) and self._map.size < 9:
            raise self.BadMapError()

        wrong_placed_puzzles = 0
        for i, puzzle in enumerate(self.flat_map):
            if puzzle == 0 and i + 1 != len(self.flat_map):
                continue
            elif puzzle != i + 1:
                wrong_placed_puzzles += 1

        return wrong_placed_puzzles

    # @property
    # def terminal_state(self):
    #     np.append(np.arange(1, dimension ** 2), 0)
    #     return

    @property
    def is_terminate(self) -> bool:
        """
        Check if all puzzles at its places
        :return: bool
        """
        return self.h == 0

    @property
    def f(self) -> int:
        return self.g + self.h


# TODO: Do we need Dequee?
class TState(Deque):

    @staticmethod
    def neighbours(current_state: super) -> super:
        raise NotImplementedError()

    @staticmethod
    def distance(a, b) -> int:
        raise NotImplementedError()

    @property
    def h(self):
        raise NotImplementedError()

    @property
    def is_terminate(self):
        raise NotImplementedError()

    @property
    def min_state(self) -> State:
        min_state = self[0]
        for elem in self:
            if elem.f < min_state.f:
                min_state = elem
        return min_state

    @staticmethod
    def reverse_to_head(state: State) -> iter:
        while state:
            yield state
            state = state.parent

    def __contains__(self, item: State) -> bool:
        matches = (True for state in self if item == state)
        return next(matches, False)


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

        # Previous code
        # wrong_placed_puzzles = 0
        # for i, puzzle in enumerate(node.flat_map):
        #     if puzzle == 0 and i + 1 != len(node.flat_map):
        #         continue
        #     elif puzzle != i + 1:
        #         wrong_placed_puzzles += 1
        return wrong_placed_puzzles

    @staticmethod
    def heuristic_manhattan(node: State) -> int:
        """
        Returns manhattan distance of all puzzles compare with terminal state
        abs(cur(i,j) - target(i,j))
        """
        # total_distance =
        total_sum = 0
        for indx_pair, value in np.ndenumerate(node._map):
            # TODO: Compare with indexes of terminal state
            pass

    @staticmethod
    def is_terminate(state: State) -> bool:
        return state.h == 0

    @staticmethod
    def neignbours(elem: State) -> TState:
        """
        Should return set of app states that can be neighbours to current
        map state (shift 0 right/left/up/down)
        """
        raise NotImplementedError()

    @staticmethod
    def h(state: State, model: str) -> int:
        """
        Should count heuristics according to chosen model
        :param state: state obj
        :param model: model name
        :return: int
        """
        raise NotImplementedError()

    @staticmethod
    def distance(first: State, second: State) -> float:
        """
        This is a simple case. So distance between each state is 1
        :return: 1.0
        """
        # raise NotImplementedError()
        return 1.0


if __name__ == '__main__':
    arr = np.ones([3, 3])
    state = State(arr)
    print(state)

    # TODO: Read from file, get map and create StartState
    read_map = arr

    start_state = State(read_map)

    _open = TState()
    _close = TState()
    _open.append(start_state)

    # next_state = State(read_map, parent=start_state)
    # _open.append(next_state)
    # print('min state: ', _open.min_state)

    while _open:
        min_state = _open.min_state
        if Rule.is_terminate(min_state):
            solution = TState(elem for elem in _open.reverse_to_head(min_state))
            solution.reverse()  # now it is solution
            exit(str(solution))
        _open.remove(min_state)
        _close.append(min_state)

        neighbours = Rule.neignbours(min_state)  # OR neighbours = min_state.all_neighbours
        for neighbour in neighbours:
            g = min_state.g + Rule.distance(min_state, neighbour)

            if neighbours in _close:
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
