import time
import argparse
from generator import generate_puzzle
from rules import Rule
from map_reader import NPuzzlesMap
from queues import StatePQueue, StateDQueue
from utils import is_solvable, get_size_comlexity


if __name__ == '__main__':
    generator = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    generator.add_argument('-G', '--generate', type=int, help="Size of the puzzle's side. Must be >= 3.", default=3,
                           dest='size')
    generator.add_argument("-s", "--solvable", action="store_true", default=False,
                           help="Forces generation of a solvable puzzle. Overrides -u.")
    generator.add_argument("-u", "--unsolvable", action="store_true", default=False,
                           help="Forces generation of an unsolvable puzzle")
    generator.add_argument("-i", "--iterations", type=int, default=10000, help="Number of passes")

    parser = argparse.ArgumentParser(parents=[generator], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--file', default='', type=str, help='Enter a path to file with puzzle',)
    parser.add_argument('-H', '--heuristics', choices=['M', 'ML', 'H', 'E', 'D'], default='M',
                        dest='heuristics',
                        help='''Choose one of heuristics to solve the puzzle.
M - for Manhattan distance.
ML - for Manhattan distance + Linear conflict.
H - for Hemming distance.
E - for Euclidean distance.
D - for Diagonal distance.
Default value is M''')

    args = parser.parse_args()

    start_time = time.time()
    Rule.choose_heuristics(args.heuristics)
    if args.file:
        npazzle = NPuzzlesMap.from_file(filename=args.file)
    else:
        string_puzzle = generate_puzzle(args)
        npazzle = NPuzzlesMap.from_string(string_puzzle)

    initial_state = npazzle.initial_state
    initial_state.f = initial_state.g + Rule.heuristics(initial_state)

    terminal_state = npazzle.terminal_state

    _open = StatePQueue(maxsize=8)
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

        params['size_complexity'] = get_size_comlexity(_open, _close)
