import pymorphy2
from static.static import IP
import elasticsearch

class CandidatesSelector():

    def __init__(self, es_host, candidates_limit: int = 5):
        self._morph = pymorphy2.MorphAnalyzer(lang='ru')
        self.candidates_limit = candidates_limit
        self.es = elasticsearch.Elasticsearch(es_host)

        self.index = "wikipedia-ru"
        self.type = "item"

    def _entity_normalizer(self, entitys_list: list):
        normalized_entitys_list = []
        for entity in entitys_list:
            normalized_entitys_list.append(self._morph.parse(entity)[0].normal_form)
        return normalized_entitys_list

    def _get_candidates_from_elastic(self, entitys_list: list):
        entity_candidates = []
        for entity in entitys_list:
            body = {
                "query": {
                    "match": {
                        "labels.ru": entity
                    }
                }
            }
            entity_candidates.append([self.es.search(index=self.index, doc_type=self.type,\
                                                    body=body, size=self.candidates_limit)])

        return entity_candidates


    def get_candidates(self, sentences_entitys_list: list):
        sentence_candidates_raw = []
        sentence_candidates = []

        for sentence_entitys in sentences_entitys_list:
            sentence_candidates_raw = self._get_candidates_from_elastic(\
                self._entity_normalizer(sentence_entitys))
            for sentence in sentence_candidates_raw:
                sentence_candidates.append(sentence[][][])



sentence_entitys_list = [["Артёма",'Организация объединённых наций'],['МВД']]
candidate_selector = CandidatesSelector(IP)
candidate_selector.get_candidates(sentence_entitys_list)
