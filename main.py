import time
from generator import generate_puzzle
from rules import Rule
from map_reader import NPuzzlesMap
from queues import StatePQueue, StateDQueue
from heuristics import PuzzleHeuristic
from utils import is_solvable, get_size_comlexity, argument_parser


if __name__ == '__main__':

    start_time = time.time()
    args = argument_parser()

    solution_case = 'ordinary' if args.ordinary else 'snail'

    if args.file:
        npazzle = NPuzzlesMap.from_file(solution_case, filename=args.file)
    else:
        string_puzzle = generate_puzzle(args, solution_case)
        npazzle = NPuzzlesMap.from_string(solution_case, string_puzzle)

    if args.greedy and args.unicost:
        print("Uniform cost and Greedy searches don't work together! Use -h option for help.")
        exit(0)

    heuristics = PuzzleHeuristic().get_heuristic(args.heuristics, args.unicost)

    initial_state = npazzle.initial_state
    initial_state.f = initial_state.g + heuristics.get_total_h(initial_state)

    terminal_state = npazzle.terminal_state

    _open = StatePQueue(maxsize=args.q_size)
    _close = StateDQueue()
    _open.put_nowait(initial_state)
    # print(initial_state)
    print('solavble?', is_solvable(initial_state._map, dimension=initial_state._map.shape[1]))

    params = dict.fromkeys(['size_complexity', 'time_complexity', 'moves_amount'], 0)
    params['size_complexity'] = get_size_comlexity(_open, _close)

    while not _open.empty():
        min_state = _open.get_nowait()
        if min_state == terminal_state:
            solution = StateDQueue(elem for elem in _close.reverse_to_head(min_state))
            solution.reverse()
            params['moves_amount'] = len(solution)
            end_time = time.time()
            delta = end_time - start_time
            print(str(solution))
            print('time complexity=', _open.time_complexity)
            print('size complexity=', params.get('size_complexity'))
            print('seconds: ', delta)
            # print('open', _open.qsize())
            print(f'Moves: {params.get("moves_amount")}')
            try:
                solution.to_file('res.json')
            except:
                pass
            # exit(str(solution))
            exit()

        _close.append(min_state)
        neighbours = Rule.neighbours(min_state)

        for neighbour in neighbours:
            if neighbour in _close:
                continue

            g = min_state.g + Rule.distance(args.greedy)

            if neighbour not in _open:
                neighbour.parent = min_state
                neighbour.set_metrics(g=g, heuristic=heuristics.get_total_h)
                _open.put_nowait(neighbour)
            elif g <= neighbour.g:
                i = _open.queue.index(neighbour)
                neighbour = _open.queue[i]
                neighbour.parent = min_state
                neighbour.set_metrics(g=g, heuristic=heuristics.get_total_h)

        params['size_complexity'] = get_size_comlexity(_open, _close)
