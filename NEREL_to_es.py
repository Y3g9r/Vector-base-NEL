import argparse
import elasticsearch
from elasticsearch import helpers




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help='Input bz2 file', type=str, default="latest-all.json.bz2")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dump(args)