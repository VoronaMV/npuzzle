from abstracts import StateDQueueABC, StatePQueueABC


class StateDQueue(StateDQueueABC):

    def __init__(self, *args, **kwargs):
        self.appends_amount = 0
        super().__init__(*args, **kwargs)

    @property
    def time_complexity(self):
        return self.appends_amount

    @staticmethod
    def reverse_to_head(state) -> iter:
        while state:
            yield state
            state = state.parent

    def append(self, item):
        self.appends_amount += 1
        return super().append(item)

    def __contains__(self, item) -> bool:
        matches = (True for state in self if item == state)
        return next(matches, False)

    def __str__(self):
        # TODO: Change it
        res = ''
        for elem in self:
            res += str(elem) + '\n\n'
        return res


class StatePQueue(StatePQueueABC):

    def __contains__(self, item) -> bool:
        matches = (True for state in self.queue if item == state)
        return next(matches, False)

    def __str__(self):
        # TODO: Change it
        res = ''
        for elem in self.queue:
            res += str(elem) + '\n\n'
        return res
