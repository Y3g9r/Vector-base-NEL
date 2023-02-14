import csv
import elasticsearch
import static

class Elastic:

    def __init__(self, es_host, max_threads=100):
        self.es = elasticsearch.Elasticsearch(es_host)
        self.max_threads = max_threads
        self.index_text = "wikipedia-ru-text"
        self.index_def = "wikipedia-ru"
        self.text_type = "text"
        self.def_type = "item"

    def get_nn_data(self):
        body_text = {
            "query": {
                "match_all": {}
            }
        }

        results = self.es.search(index=self.index_text, doc_type=self.text_type, body=body_text)
        founded_count = results["hits"]["total"]

        results_text = self.es.search(index=self.index_text, doc_type=self.text_type, body=body_text, size=founded_count)

        csv_data = []
        for result_text in results_text["hits"]["hits"]:
            current_id = result_text["_id"]
            body_def = {
                "query": {
                    "match": {
                        "_id": current_id
                    }
                }
            }
            result_data = self.es.search(index=self.index_def, doc_type=self.def_type, body=body_def)
            if result_data["hits"]["total"] != 0:
                temp_data = result_data["hits"]["hits"][0]["_source"]
                if "ru" in temp_data["descriptions"]:
                    csv_data.append([result_text["_source"]["text"], result_text["_source"]["position"], temp_data["descriptions"]["ru"]])

        return csv_data



es = Elastic(static.IP)
csv_data = es.get_nn_data()

header = ['text', 'position', 'definition']

with open('nn_data.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)

    for data in csv_data:
        writer.writerow(data)
