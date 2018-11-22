from abstracts import StateABC


class State(StateABC):

    terminal_map = None

    def shift_empty_puzzle(self, direction) -> tuple:
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
