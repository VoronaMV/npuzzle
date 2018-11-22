import argparse


class AParser:

    @staticmethod
    def argument_parser():
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
        return args
