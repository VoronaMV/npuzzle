import time
from generator import generate_puzzle
from rules import Rule
from map_reader import NPuzzlesMap
from queues import StatePQueue, StateDQueue
from utils import is_solvable, argument_parser


if __name__ == '__main__':

    start_time = time.time()
    args = argument_parser()

    Rule.choose_heuristics(args.heuristics)
    solution_case = 'ordinary' if args.ordinary else 'snail'

    if args.file:
        npazzle = NPuzzlesMap.from_file(solution_case, filename=args.file)
    else:
        string_puzzle = generate_puzzle(args)
        npazzle = NPuzzlesMap.from_string(solution_case, string_puzzle)

    initial_state = npazzle.initial_state
    initial_state.f = initial_state.g + Rule.heuristics(initial_state)

    terminal_state = npazzle.terminal_state

    # The bigger size = the shorter way, but longer time
    # 8 was very fast
    _open = StatePQueue(maxsize=args.q_size) # 50 was good for 4*4 3 на 3 тоже) Map 4*4 inversion=45 the best was max_size=15 (inv number=6 - time=0.4 sec)
    _close = StateDQueue()
    _open.put_nowait(initial_state)
    print(initial_state)
    print('solavble?', is_solvable(initial_state._map, dimension=initial_state._map.shape[1]))

    while not _open.empty():
        min_state = _open.get_nowait()
        if min_state == terminal_state:
            solution = StateDQueue(elem for elem in _close.reverse_to_head(min_state))
            solution.reverse()  # now it is solution
            moves_number = len(solution)
            end_time = time.time()
            delta = end_time - start_time
            print(str(solution))
            print('seconds: ', delta)
            print('open', _open.qsize())
            print(f'Moves: {moves_number}')
            solution.to_file('res.json')
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
                neighbour.set_metrics(g=g, heuristic=Rule.heuristics)
                _open.put_nowait(neighbour)
            elif g <= neighbour.g:
                i = _open.queue.index(neighbour)
                neighbour = _open.queue[i]
                neighbour.parent = min_state
                neighbour.set_metrics(g=g, heuristic=Rule.heuristics)
