import os

import argparse

import elasticsearch
from elasticsearch import helpers

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())

def read_ann_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())

    return id_pos


def dump(args):
    os.chdir(args.input)
    for file in os.listdir():
        if file.endswith(".ann"):
            try:
                text_file = file.rstrip(".ann")+".txt"
                text_data = read_txt_file(text_file)
                text_meta = read_ann_file(file)
            except Exception as e:
                print(e)
                continue
        if file.endswith(".txt"):
            try:
                ann_file = file.rstrip(".txt") + ".ann"
                text_meta = read_ann_file(ann_file)
                text_data = read_txt_file(file)
            except Exception as e:
                print(e)
                continue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help='Input directory name with NEREL data', type=str, default="train")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dump(args)