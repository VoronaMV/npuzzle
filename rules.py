import math
import numpy as np
from state import State


class Rule:

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
    def distance(greedy=None) -> float:
        """
        This is a simple case. So distance between each state is 1
        :return: 1.0
        """
        if greedy:
            return 0
        return 1.0
