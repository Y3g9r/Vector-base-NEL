import csv
import elasticsearch
from static.static import IP
from tqdm import tqdm

class Elastic:

    def __init__(self, es_host, max_threads=100):
        self.es = elasticsearch.Elasticsearch(es_host)
        self.max_threads = max_threads
        self.index_text = "wikipedia-ru-text"
        self.index_def = "wikipedia-ru"
        self.text_type = "text"
        self.def_type = "item"

        self.passed_negative_defs = []

    def get_nn_data(self) -> list:
        body_text = {
            "query": {
                "match_all": {}
            }
        }

        results = self.es.search(index=self.index_text, doc_type=self.text_type, body=body_text)
        founded_count = results["hits"]["total"]

        results_text = self.es.search(index=self.index_text, doc_type=self.text_type, body=body_text, size=founded_count)

        csv_data = []
        for result_text in tqdm(results_text["hits"]["hits"]):
            current_id = result_text["_id"]
            body_def = {
                "query": {
                    "bool": {
                        "must":[{
                            "match": {
                                "_id": current_id
                            }
                        }],
                        "must_not":[{
                            "match": {
                                "descriptions.ru": "страница значений"
                            }
                        }]
                    }
                }
            }
            result_data = self.es.search(index=self.index_def, doc_type=self.def_type, body=body_def)
            if result_data["hits"]["total"] != 0:
                temp_data = result_data["hits"]["hits"][0]["_source"]
                record = es._get_negative_record(temp_data["labels"]["ru"], current_id)
                if "ru" in temp_data["descriptions"]:
                    csv_data.append([result_text["_source"]["text"], result_text["_source"]["position"],
                                     temp_data["descriptions"]["ru"], 1])
                    csv_data.append([result_text["_source"]["text"], result_text["_source"]["position"],
                                     record, 0])

        return csv_data

    def _get_negative_record(self, label_ru: str, current_id: int):
        body_def_negative = {
            "query": {
                "bool": {
                    "must": [{
                        "match": {
                            "labels.ru": label_ru
                        }
                    },
                        {
                            "exists": {
                                "field": "descriptions.ru"
                            }
                        }],
                    "must_not": [
                    {"match": {
                      "_id": current_id
                    }},
                        {"match": {
                            "descriptions.ru": "страница значений астероид"
                        }}
                  ]
                }
            }
        }

        body_random_doc = {
          "size":1,
          "query": {

            "function_score": {
              "query": {
                "bool": {
                  "must": [{
                    "exists": {
                          "field": "descriptions.ru"
                      }
                    }],
                    "must_not": [
                        {"match": {
                            "descriptions.ru": "страница значений астероид"
                        }}
                    ]
                }
              },
              "functions": [
                {
                  "random_score": {
                  }
                }
              ]
            }
          }
        }

        result_data = self.es.search(index=self.index_def, doc_type=self.def_type, body=body_def_negative)

        if result_data["hits"]["total"] == 0 or \
            result_data["hits"]["hits"][0]["_source"]["descriptions"]["ru"] in self.passed_negative_defs:
            result_data = self.es.search(index=self.index_def, doc_type=self.def_type, body=body_random_doc)
            while result_data["hits"]["hits"][0]["_id"] == current_id or\
                    result_data["hits"]["hits"][0]["_source"]["descriptions"]["ru"] in self.passed_negative_defs:
                result_data = self.es.search(index=self.index_def, doc_type=self.def_type, body=body_random_doc)

        def_negative = result_data["hits"]["hits"][0]["_source"]["descriptions"]["ru"]
        self.passed_negative_defs.append(def_negative)

        return def_negative


es = Elastic(IP)
csv_data = es.get_nn_data()

header = ["text", "position", "definition", "label"]

with open('nn_data.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)

    for data in csv_data:
        writer.writerow(data)
