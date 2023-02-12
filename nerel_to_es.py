import os

import argparse

import elasticsearch
from elasticsearch import helpers

def read_txt_file(file_path):
    text_data = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) is not 0:
                text_data.append(line)

    return text_data


def read_ann_file(file_path):
    text_meta = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) is not 0:
                parts_of_line = line.split(' ')
                if line.startswith("T"):
                    pos = []
                    for info in enumerate(parts_of_line):
                        if parts_of_line[info].isdigit():
                            pos.append(int(parts_of_line[info], parts_of_line[info+1]))
                        meta_data[parts_of_line[0]] = [pos]
                        meta_data[parts_of_line[0]].append[" ".join(parts_of_line[info+2:])]
                        break
                if line.startswith("N"):
                    for info in enumerate(parts_of_line):
                        if parts_of_line[info].startswith("T"):
                            if parts_of_line[info+1].startswith("Wikidata:") and \
                                    parts_of_line[info+1].lstrip("Wikidata:") is not "NULL":
                                meta_data[parts_of_line[info]] = \
                                    [meta_data[parts_of_line[info]], parts_of_line[info+1].lstrip("Wikidata:")]
                                break

    return text_meta


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
                text_data = read_txt_file(file)
                text_meta = read_ann_file(ann_file)
            except Exception as e:
                print(e)
                continue
        for record in text_meta:
            for record_text in text_data:
                if record_text[record[0][0]:record[0][1]] == record[1]:
                    pass



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help='Input directory name with NEREL data', type=str, default="train")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dump(args)