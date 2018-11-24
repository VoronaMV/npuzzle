#!/usr/bin/env python

import random
import numpy as np
from utils import is_solvable


def make_puzzle(s, solvable, iterations, solution_case):
    def swap_empty(p):
        idx = p.index(0)
        poss = []
        if idx % s > 0:
            poss.append(idx - 1)
        if idx % s < s - 1:
            poss.append(idx + 1)
        if idx / s > 0:
            poss.append(idx - s)
        if idx / s < s - 1:
            poss.append(idx + s)
        swi = random.choice(poss)
        p[idx] = p[swi]
        p[swi] = 0

    p = make_goal(s)
    for i in range(iterations):
        swap_empty(p)
    if not solvable:
        if p[0] == 0 or p[1] == 0:
            p[-1], p[-2] = p[-2], p[-1]
        else:
            p[0], p[1] = p[1], p[0]
    np_arr = np.array(p)
    np_arr = np_arr.reshape((s, s))
    if solvable:
        if is_solvable(np_arr, s, solution_case):
            return p
        else:
            return make_puzzle(s, solvable, iterations, solution_case)
    # else:
    return p


def make_goal(s):
    ts = s * s
    puzzle = [-1 for i in range(ts)]
    cur = 1
    x = 0
    ix = 1
    y = 0
    iy = 0
    while True:
        puzzle[x + y * s] = cur
        if cur == 0:
            break
        cur += 1
        if x + ix == s or x + ix < 0 or (ix != 0 and puzzle[x + ix + y * s] != -1):
            iy = ix
            ix = 0
        elif y + iy == s or y + iy < 0 or (iy != 0 and puzzle[x + (y + iy) * s] != -1):
            ix = -iy
            iy = 0
        x += ix
        y += iy
        if cur == s * s:
            cur = 0

    return puzzle


def stringify_map(map, map_size):
    space_per_element = len(str(map_size ** 2))
    stringified_map = ''
    for y in range(map_size):
        for x in range(map_size):
            stringified_map += str(map[x + y * map_size]).rjust(space_per_element) + ' '
        stringified_map += '\n'
    return stringified_map


def generate_puzzle(args, solution_case):
    if args.solvable and args.unsolvable:
        print("Can't be both solvable AND unsolvable, dummy !")
        exit(1)

    if args.size < 3:
        print("Can't generate a puzzle with size lower than 2. It says so in the help. Dummy.")
        exit(1)

    if args.solvable:
        solvable = True
    elif args.unsolvable:
        solvable = False
    else:
        solvable = random.choice([True, False])

    map_size = args.size
    puzzle = make_puzzle(map_size, solvable=solvable, iterations=args.iterations, solution_case=solution_case)

    print(args.solvable)

    is_solvable_str = "solvable" if solvable else "unsolvable"
    print(f"# This puzzle is {is_solvable_str}")

    stringified_map = str(map_size) + '\n' + stringify_map(puzzle, map_size)
    print(stringified_map, end='')
    return stringified_map
