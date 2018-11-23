import numpy as np
from hashlib import sha1
from abc import ABCMeta, abstractmethod
from queue import PriorityQueue
from typing import Deque
from utils import TERMINAL_STATES


class StateABC(metaclass=ABCMeta):

    terminal_map = None

    def __init__(self, data: np.ndarray, parent=None, heuristic: callable=None):
        if not isinstance(data, np.ndarray) and data.size < 9:
            raise StateABC.BadMapError()
        self._map = data.astype(int)
        self.flat_map = self._map.flatten()
        self.parent = parent
        self.g = parent.g + 1 if parent else 0
        # TODO: Make better way
        # if self.terminal_map is None:
        #     dimension, _ = self._map.shape
        #     self.terminal_map = TERMINAL_STATES.get(dimension)

        self.empty_puzzle_coord = self.empty_element_coordinates(self._map)
        if heuristic:
            self.f = self.g + heuristic(self)
        else:
            self.f = None
        self.hash = sha1(self._map).hexdigest()

    @staticmethod
    def empty_element_coordinates(_map: np.ndarray) -> tuple:
        for indx_pair, elem in np.ndenumerate(_map):
            if elem == 0:
                return indx_pair

    class UnknownInstanceError(Exception):
        def __init__(self, message='Unknown class instance', error=None):
            super().__init__(message)
            self.error = error

    class BadMapError(Exception):
        def __init__(self, message='Bad map', error=None):
            super().__init__(message)
            self.error = error

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        pass

    @abstractmethod
    def __le__(self, other) -> bool:
        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        pass

    @abstractmethod
    def __ge__(self, other) -> bool:
        pass

    @abstractmethod
    def shift_empty_puzzle(self, direction: str) -> tuple:
        pass

    @abstractmethod
    def set_metrics(self, heuristic: callable, g: int=None) -> None:
        pass


class StatePQueueABC(PriorityQueue, metaclass=ABCMeta):

    time_complexity = 0

    @abstractmethod
    def __contains__(self, item: StateABC) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def put_nowait(self, item):
        qsize = self.qsize()
        if self.maxsize and self.maxsize == qsize:
            self.time_complexity = qsize
            self.pop_max()
        else:
            self.time_complexity = qsize + 1
        return super().put_nowait(item)

    def pop_max(self):
        max_element = max(self.queue)
        if not max_element:
            return
        max_index = self.queue.index(max_element)
        return self.queue.pop(max_index)


class StateDQueueABC(Deque, metaclass=ABCMeta):

    @abstractmethod
    def time_complexity(self) -> int:
        pass

    @staticmethod
    @abstractmethod
    def reverse_to_head(state: StateABC) -> iter:
        pass

    @abstractmethod
    def __contains__(self, item: StateABC) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
