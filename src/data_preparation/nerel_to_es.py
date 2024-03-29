import os

import argparse

import elasticsearch
from elasticsearch import helpers
import concurrent.futures
import re
import datetime
from tqdm import tqdm

import static


class ElasticDumper:

    def __init__(self, es_host, max_threads=100):
        self.es = elasticsearch.Elasticsearch(es_host)
        self.max_threads = max_threads

    def write_records(self, records_list):
        records = self._prepare_actions(records_list)
        helpers.bulk(self.es, records)

    def _prepare_actions(self, records_list):

        if len(records_list) > self.max_threads:
            num_threads = self.max_threads
        else:
            num_threads = len(records_list)

        futures = []
        records = []
        with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
            for record in records_list:
                futures.append(executor.submit(self.__prepare_action, record=record))
        for future in futures:
            try:
                res = future.result()
                if res:
                    records.append(res)
            except Exception:
                print(traceback.format_exc())
                pass
        return records

    def __prepare_action(self, record):
        action = {
            "_index": "wikipedia-ru-text",
            "_type": "text",
            "_id": record[0],
            "_source": {
                "timestamp": datetime.datetime.now(),
                "position": record[2],
                "text": record[1]
            }
        }
        return action


def read_txt_file(file_path):
    text_data = []

    with open(file_path, encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            if len(line) != 0 and line != '\n':
                text_data.append(line.rstrip('\n'))

    return text_data


def read_ann_file(file_path):
    text_meta = {}
    with open(file_path, encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            if len(line) != 0:
                parts_of_line = re.sub(r'\t', ' ', line) .split(' ')
                parts_of_line[-1] = parts_of_line[-1].rstrip('\n')
                if line.startswith("T"):
                    pos = []
                    for i, info in enumerate(parts_of_line):
                        if parts_of_line[i].isdigit() and parts_of_line[i+1].isdigit():
                            pos.append(int(parts_of_line[i]))
                            pos.append(int(parts_of_line[i+1]))
                        else:
                            continue
                        text_meta[parts_of_line[0]] = [pos]
                        text_meta[parts_of_line[0]].append(" ".join(parts_of_line[i+2:]))
                        break

                if line.startswith("N"):
                    for i, info in enumerate(parts_of_line):
                        if info.startswith("T"):
                            if parts_of_line[i+1].startswith("Wikidata:") and \
                                    parts_of_line[i+1].lstrip("Wikidata:") != "NULL" and info in text_meta:
                                text_meta[info].append(parts_of_line[i+1].lstrip("Wikidata:"))
                                break

    processed_text_meta = {key:val for key,val in text_meta.items() if len(val) == 3 }
    return processed_text_meta


def dump(args):
    os.chdir(args.input)

    es_client = ElasticDumper(static.IP)
    passed_files = []
    for file in tqdm(os.listdir()):
        if file not in passed_files and file.endswith(".ann"):
            try:
                text_file = file.rstrip(".ann")+".txt"
                text_data = read_txt_file(text_file)
                text_meta = read_ann_file(file)
                passed_files.append(file)
                passed_files.append(text_file)
            except Exception as e:
                print(e)
                continue
        if file not in passed_files and file.endswith(".txt"):
            try:
                ann_file = file.rstrip(".txt") + ".ann"
                text_data = read_txt_file(file)
                text_meta = read_ann_file(ann_file)
                passed_files.append(file)
                passed_files.append(ann_file)
            except Exception as e:
                print(e)
                continue

        records = []
        for record in text_meta.values():
            for record_text in text_data:
                if record_text[record[0][0]:record[0][1]] == record[1]:
                    records.append([record[2], record_text, record[0]])
                    break
        if len(records) != 0:
            es_client.write_records(records)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help='Input directory name with NEREL data', type=str, default="train")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dump(args)