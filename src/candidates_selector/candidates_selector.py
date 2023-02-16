import pymorphy2
from static.static import IP
import elasticsearch

class CandidatesSelector():

    def __init__(self, es_host, candidates_limit: int = 5):
        print(IP)
        self._morph = pymorphy2.MorphAnalyzer(lang='ru')
        self.candidates_limit = candidates_limit
        self.es = elasticsearch.Elasticsearch(es_host)

        self.index = "wikipedia-ru"
        self.type = "item"

    def _entity_normalizer(self, entity: str):
        return self._morph.parse(entity)[0].normal_form

    def _get_candidates_from_elastic(self, entity: str):
        body = {
            "query": {
                "match": {
                    "labels.ru": entity
                }
            }
        }

        return self.es.search(index=self.index, doc_type=self.type, body=body, size = self.candidates_limit)


    def get_candidates(self, sentences_entitys_list: list):
        sentence_candidates_raw = []
        sentence_candidates = []

        for sentence_entitys in sentences_entitys_list:
            for entity in sentence_entitys:
                sentence_candidates_raw = self._get_candidates_from_elastic(\
                    self._entity_normalizer(entity))
                for sentence in sentence_candidates_raw:



sentence_entitys_list = [["Артёма",'Организация объединённых наций'],['МВД']]
candidate_selector = CandidatesSelector(IP)
candidate_selector.get_candidates(sentence_entitys_list)
