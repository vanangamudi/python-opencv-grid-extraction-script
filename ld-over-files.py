from glob import glob
import sys

from line_detector import process, write_shapes
from pprint import pprint, pformat

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Grid-segmenter')
    
    parser.add_argument('-d','--prefix-dir',
                        help='path to the image file',
                        default='output', dest='prefix')

    parser.add_argument('-s','--size',
                        help='size of the resulting shape',
                        default=120, dest='size')
    
    parser.add_argument('-v', '--verbose',
                        help='shows all the grid overlayed in input image',
                        action='store_true', default=False, dest='verbose')

    parser.add_argument('-V', '--very-verbose',
                        help='shows all the pieces of the characters',
                        action='store_true', default=False, dest='very_verbose')
        
    args = parser.parse_args()
    count = 0
    total_counts = {}
    for filepath in glob('../raw_data/sheets/brahmi_data/*.jpg'):
        try:
            args.filepath = filepath
            total_count, shapes = process(args, filepath)
            total_counts[filepath] = total_count

            write_shapes(args, shapes)

        except KeyboardInterrupt:
            exit(0)
        except:
            print(sys.exc_info())
            count += 1


    print(count, ' exceptions raised')

    pprint(total_counts)
